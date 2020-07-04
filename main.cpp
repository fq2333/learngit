#include <iostream>
#include <time.h>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>

#include "loadstl_paja.h"
#include "bvh_paja.h"
#include "camera_paja.h"
#include "rand_paja.h"
#include "raytrace_paja.h"
#include "bmp_paja.h"
#include "enviro_paja.h"
#include "star_background.h"
#include "defines_paja.h"

using namespace std;

int main() {

	clock_t startTime, endTime;

	//create object scenes
	stlscenes ssce(3);
	//ssce.filelist[0] = stlfile("./mod/oneguonew.STL", 0);
	ssce.filelist[0] = stlfile("./mod/a_zhuti.STL", 0);
	ssce.filelist[1] = stlfile("./mod/a_qita.STL", 1);
	ssce.filelist[2] = stlfile("./mod/a_fanban.STL", 2);
	bvhscenes bsce;
	bsce.get_root(ssce, "./bvh/1.data");
	//create enviroment
	source* sor;
	cudaMallocManaged((void**)&sor, sizeof(source));
	*sor = source(vec3f(1.f, 0.6f, 0.8f), 100e9f, 1e9f, 3.8e10f, 0.3f);

#if DRAW_BVH
	startTime = clock();
	drawbvh("./mod/_.obj", bsce.bvhlist);
	endTime = clock();
	cout << "bvh draw time : " << endTime - startTime << endl;
#endif

	//create camera
	camera* cam;
	cudaMallocManaged((void**)&cam, sizeof(camera));
	*cam = camera(vec3f(1.f, 0.2f, 0.1f), 2.5e3f, vec3f(0.f, 0.f, 20.f), vec3f(0.f, 0.f, 1.f));//�۲췽����࣬���λ�ã��ο��춥����
	cam->set_baspara(2048, 2048, 8e-3f);//��������Ԫ�ߴ�
	cam->set_optpara(20.f, 0.f, 5.f, 0.85f, 0.55e-6f);//���ࡢ�뽹���ھ�����ѧ͸���ʡ����Ĳ���
	cam->set_elcpara(30.f, 1e5f, vec3f(0.25f), 2.0f);//����ʱ�䡢���µ�ɡ�����Ч�ʡ��˶��ٶȡ�����
	cam->set_noise(25.f, 0.f, 0.f);//�¶ȡ���������������������
	//create objpos
	vec3f* objpos;
	cudaMallocManaged((void**)&objpos, sizeof(vec3f) * cam->w * cam->h);

	startTime = clock();
	camera_init_gpu(cam, objpos);
	endTime = clock();
	cout << "camera init time : " << endTime - startTime << endl;

	//create rand states
	curandState* randst;
	cudaMallocManaged((void**)&randst, sizeof(curandState) * cam->w * cam->h);
	int t0 = ((int)time(NULL)) % 8;

	startTime = clock();
	rand_init_gpu(randst, cam, t0);
	endTime = clock();
	cout << "rand init time : " << endTime - startTime << endl;

#if RAND_TEST
	startTime = clock();
	rand_test_gpu(randst, cam, 200);
	endTime = clock();
	cout << "rand test time : " << endTime - startTime << endl;
#endif

	//create frame buffer
	vec3f* fb;
	cudaMallocManaged((void**)&fb, sizeof(vec3f) * cam->w * cam->h);


	//������ͼ
	Star_Background* SB;
	cudaMallocManaged((void**)&SB, sizeof(Star_Background));
	*SB = Star_Background(MAGNITUDE, "./mod/HYG.dat");
	vec4f* Star_Data;
	cudaMallocManaged((void**)&Star_Data, sizeof(vec4f) * 200000);
	int* Star_Data_Num;
	cudaMallocManaged((void**)&Star_Data_Num, sizeof(int));
	*Star_Data_Num = SAOread(SB->str, Star_Data);
	startTime = clock();
	SAO_process_gpu(Star_Data, Star_Data_Num, cam, sor, SB, fb);
	endTime = clock();
	cout << "Star Background test time : " << endTime - startTime << "\n" << endl;
	//create_bmp("./out/03.bmp", fb, cam->w, cam->h);




	//ray trace
	startTime = clock();
	raytrace_test_gpu(bsce.bvhlist, cam, objpos, sor, randst, fb);
	endTime = clock();
	cout << "ray trace test time : " << endTime - startTime << "\n" << endl;

	//out put data
	create_bmp("./out/04.bmp", fb, cam->w, cam->h);

#if CONTINUE
	//continue rt
	int conti;
	cout << "continue? (1/0) : ";
	cin >> conti;
	while (conti) {
		cout << "reset cam? (1/0) : ";
		cin >> conti;
		if (conti) {
			cout << "reset options:" << endl;
			cout << "1.cam_dir\t" << "2.cam_len\t" << "3.cam_lookat" << endl;
			cout << "4.cam_updir\t" << "5.pixsize\t" << "6.foclen" << endl;
			cout << "7.defoclen\t" << "8.dsize\t\t" << "9.speed" << endl;
			cout << "10.exptime\t" << "11.emax\t\t" << "12.ita" << endl;
			cout << "13.tep\t\t" << "14.dnoise\t" << "15.rnoise" << endl;
			cout << "16.sor_dir\t" << "17.sor_eng" << endl;
		}
		bool init = false;
		while (conti) {
			float x, y, z;
			int selc;
			cout << "select (1-17) : ";
			cin >> selc;
			switch (selc)
			{
			case 1:
				cout << "input cam_dir(x,y,z) : ";//�������ָ��
				cin >> x >> y >> z;
				cam->set_camdir(vec3f(x, y, z));
				init = true;
				break;
			case 2:
				cout << "input cam_len : ";//����۲�����(2e3mm)
				cin >> x;
				cout << "auto foc? (1/0) : ";//�Զ��Խ�
				cin >> selc;
				cam->set_camlen(x, selc);
				init = true;
				break;
			case 3:
				cout << "input cam_lookat(x,y,z) : ";//�۲��
				cin >> x >> y >> z;
				cout << "auto foc? (1/0) : ";
				cin >> selc;
				cam->set_camla(vec3f(x, y, z), selc);
				init = true;
				break;
			case 4:
				cout << "input cam_updir(x,y,z) : ";//�ο��춥����
				cin >> x >> y >> z;
				cam->set_camupd(vec3f(x, y, z));
				init = true;
				break;
			case 5:
				cout << "input pix size : ";//��Ԫ�ߴ�
				cin >> x;
				cam->set_pixsize(x);
				init = true;
				break;
			case 6:
				cout << "input foclen : ";//����
				cin >> x;
				cout << "auto foc? (1/0) : ";
				cin >> selc;
				cam->set_foclen(x, selc);
				init = true;
				break;
			case 7:
				cout << "input defoclen : ";//�뽹��
				cin >> x;
				cam->set_defoc(x);
				init = true;
				break;
			case 8:
				cout << "input dsize : ";//�ھ�
				cin >> x;
				cam->set_dsize(x);
				break;
			case 9:
				cout << "input speed(x,y,z) : ";//�ٶ�(mm/ms)
				cin >> x >> y >> z;
				cam->set_speed(vec3f(x, y, z));
				break;
			case 10:
				cout << "input exposing time : ";//����ʱ��exptime
				cin >> x;
				cam->set_exptime(x);
				break;
			case 11:
				cout << "input max charge : ";//���µ��
				cin >> x;
				cam->set_emax(x);
				break;
			case 12:
				cout << "input ita(x,y,z) : ";//����Ч�ʣ�0.2 0 0 ��RGB
				cin >> x >> y >> z;
				cam->set_ita(vec3f(x, y, z));
				break;
			case 13:
				cout << "input temperature : ";//�¶�
				cin >> x;
				cam->set_tep(x);
				break;
			case 14:
				cout << "input dnoise : ";//
				cin >> x;
				cam->set_dnoise(x);
				break;
			case 15:
				cout << "input rnoise : ";
				cin >> x;
				cam->set_rnoise(x);
				break;
			case 16:
				cout << "input sor_dir(x,y,z) : ";//̫������ �����ԭ��
				cin >> x >> y >> z;
				sor->set_sordir(vec3f(x, y, z));
				break;
			case 17:
				cout << "input sor_eng : ";//̫������
				cin >> x;
				sor->set_soreng(x);
				break;
			default:
				break;
			}

			cout << "continue reset? (1/0) : ";
			cin >> conti;
		}

		//camera reinit
		if (init) camera_init_gpu(cam, objpos);

		cout << "rand re-init? (1/0) : ";
		cin >> conti;
		if (conti) {
			t0 = ((int)time(NULL)) % 8;
			rand_init_gpu(randst, cam, t0);
		}

		//ray trace
		startTime = clock();
		raytrace_test_gpu(bsce.bvhlist, cam, objpos, sor, randst, fb);
		endTime = clock();
		cout << "ray trace test time : " << endTime - startTime << "\n" << endl;

		//out put data
		create_bmp("./out/01.bmp", fb, cam->w, cam->h);

		cout << "continue? (1/0) : ";
		cin >> conti;
	}
#endif

	//free mem
	cudaFree(fb);
	free_scenes(bsce);
	free_camera(cam, objpos);
	cudaFree(randst);
	cudaFree(sor);

	return 0;
}