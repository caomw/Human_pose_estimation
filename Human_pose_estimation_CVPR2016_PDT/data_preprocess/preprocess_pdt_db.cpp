// train_ONE_BLLY2.cpp : Defines the entry point for the console application.
//
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp> 
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <cstring>
#include <stdlib.h>


#define PDT_FOLDER		"/media/sda1/Study/Data/Human_pose_estimation/PDT/"
#define SAVE_FOLDER		"/media/sda1/Study/Data/Human_pose_estimation/PDT/PDT2EVAL/time_sequence/"

#define Q_WIDTH					320
#define Q_HEIGHT				240
#define BACKGROUND_DEPTH		100

using namespace std;
using namespace cv;
enum{
		ONE_HEAD		=	0,
		ONE_SDER		=	1,
		ONE_TRSO		=	2, // kinect correspondence: middle of the spine
		ONE_BLLY		=	3, // kinect correspondence: top triangular point between left and right hip
	
		LFT_SDER		=	4,
		LFT_EBOW		=	5,
		LFT_HAND		=	6,  // kinect correspondence: average of hand and wrist
		LFT_HIPS		=	7,
		LFT_KNEE		=	8,
		LFT_AKLE		=	9, // kinect correspondence: left ankle 

		RHT_SDER		=	10,
		RHT_EBOW		=	11,
		RHT_HAND		=	12,  // kinect correspondence: average of habd abd wrist
		RHT_HIPS		=	13,
		RHT_KNEE		=	14,
		RHT_AKLE		=	15, // kinect correspondence: right ankle
	
		JOINT_NUMBER	=	16
	}BODY_IDX;

inline void world2pixel( float pt[3] ){
	pt[0] = 160 + pt[0]*258.2f/pt[2];
	pt[1] = 120 + (pt[1]*258.2/pt[2]);

	if( pt[0] < 0 ){ pt[0] = 0; }
	if( pt[1] < 0 ){ pt[1] = 0; }
	if( pt[0] >= 1024  ){ pt[0] = 1024-1; }
	if( pt[1] >= 1024 ){  pt[1] = 1024-1; }
	//if( pt[0] >= Q_WIDTH  ){ pt[0] = Q_WIDTH-1; }
	//if( pt[1] >= Q_HEIGHT ){ pt[1] = Q_HEIGHT-1; }
}
inline void pixel2world( float pt[3] ){
	pt[0] = (pt[0]-160.0f)*pt[2]*0.003873f; 
	pt[1] = (pt[1]-120.0f)*pt[2]*0.003873f; 
}
inline void m4x4v4( float m[4][4], float src[4], float dst[4] ){
	dst[0] = m[0][0]*src[0] + m[0][1]*src[1] +  m[0][2]*src[2] + m[0][3]*src[3];
	dst[1] = m[1][0]*src[0] + m[1][1]*src[1] +  m[1][2]*src[2] + m[1][3]*src[3];
	dst[2] = m[2][0]*src[0] + m[2][1]*src[1] +  m[2][2]*src[2] + m[2][3]*src[3];
	dst[3] = m[3][0]*src[0] + m[3][1]*src[1] +  m[3][2]*src[2] + m[3][3]*src[3];
}
inline void pdt2eval( float m[4][4], float pt3[3] ){
	float pt4[3], ptout[4];
	pt4[0] = pt3[0]; pt4[1] = pt3[1]; pt4[2] = pt3[2]; pt4[3] = 1;
	m4x4v4( m, pt4, ptout );
	pt3[0] = ptout[0]; pt3[1] = ptout[1]; pt3[2] = ptout[2];
	world2pixel( pt3 );
}

bool load2image( cv::Mat *image, char *path, float extm[4][4]){
	bool load_successful = false;
	std::ifstream in( path);
	int px, py;
	float mx, my, mz;
	//float min[3], max[3];
	//min[0] = min[1] = min[2] = FLT_MAX;
	//max[0] = max[1] = max[2] = -FLT_MAX;

	image->setTo(255);
	while(in >> py){
		in >> px;
		in >> mx;
		in >> my;
		in >> mz;

		float wpt[4];
		wpt[0] = mx; wpt[1] = my; wpt[2] = mz; wpt[3] = 1;
		pdt2eval( extm, wpt );

		int ix, iy;
		ix = int( wpt[0] +0.5f );
		iy = int( wpt[1] +0.5f );
		int gray = int((-wpt[2]/4)*255);
		image->data[ iy*Q_WIDTH + ix ] = gray;

		load_successful = true;
	}
	in.close();
	return load_successful;
}
bool load2depth( float depth[Q_WIDTH*Q_HEIGHT], float user[Q_WIDTH*Q_HEIGHT], float body_center[3], char *path, float extm[4][4]){
	bool load_successful = false;
	std::ifstream in( path);
	int px, py;
	float mx, my, mz;
	float count[Q_WIDTH*Q_HEIGHT];
	for( int p = 0 ; p < Q_WIDTH*Q_HEIGHT ; p++){ count[p] = 0; }
	for( int p = 0 ; p < Q_WIDTH*Q_HEIGHT ; p++){ depth[p] = 0; }
	for( int p = 0 ; p < Q_WIDTH*Q_HEIGHT ; p++){ user[ p] = 0; }

	while(in >> py){
		in >> px;
		in >> mx;
		in >> my;
		in >> mz;

		float wpt[4];
		wpt[0] = mx; wpt[1] = my; wpt[2] = mz; wpt[3] = 1;
		pdt2eval( extm, wpt );

		int ix, iy;
		ix = int( wpt[0] +0.5f );
		iy = int( wpt[1] +0.5f );

		depth[ iy*Q_WIDTH + ix ] += wpt[2];
		count[ iy*Q_WIDTH + ix ] ++;

		load_successful = true;
	}
	for( int p = 0 ; p < Q_WIDTH*Q_HEIGHT ; p++){ 
		if( count[p] >= 1 ){
			user[ p] = 1;
			depth[p] /= count[p];
		}
		else{
			user[ p] = 0;
			depth[p] = BACKGROUND_DEPTH;
		}
	}
	float body_count = 0;
	body_center[0] = 0;
	body_center[1] = 0;
	body_center[2] = 0;
	for( int y = 0 ; y < Q_HEIGHT ; y++ ){
		for( int x = 0 ; x < Q_WIDTH ; x++ ){
			int p = y*Q_WIDTH + x;
			if( user[p] == 1 ){
				body_center[0] += x;
				body_center[1] += y;
				body_center[2] += depth[p];
				body_count++;
			}
		}
	}
	body_center[0] /= body_count;
	body_center[1] /= body_count;
	body_center[2] /= body_count;
	pixel2world( body_center );


	in.close();
	return load_successful;
	// also need to get BLLY_ONE from xyz center
}
bool load2joint( std::ifstream *joint_in, float p_joint[JOINT_NUMBER*3] , float extm[4][4] ){
	bool load_successful = false;
	for( int j = 0 ; j < JOINT_NUMBER*3 ; j++ ){ p_joint[j] = 0; }

	for( int idx = 1 ; idx <= 20 ; idx++ ){
		float xyz[3];
		float garbage;
		*joint_in >> xyz[0];
		*joint_in >> xyz[1];
		*joint_in >> xyz[2];
		*joint_in >> garbage;
		*joint_in >> garbage;
		*joint_in >> garbage;
		if( *joint_in >> garbage ){ load_successful = true; }
		if( idx == 2  ){ p_joint[ONE_BLLY*3+0] = xyz[0]; p_joint[ONE_BLLY*3+1] = xyz[1]; p_joint[ONE_BLLY*3+2] = xyz[2];  }
		if( idx == 3  ){ p_joint[ONE_SDER*3+0] = xyz[0]; p_joint[ONE_SDER*3+1] = xyz[1]; p_joint[ONE_SDER*3+2] = xyz[2];  }
		if( idx == 4  ){ p_joint[ONE_HEAD*3+0] = xyz[0]; p_joint[ONE_HEAD*3+1] = xyz[1]; p_joint[ONE_HEAD*3+2] = xyz[2];  }
		if( idx == 5  ){ p_joint[LFT_SDER*3+0] = xyz[0]; p_joint[LFT_SDER*3+1] = xyz[1]; p_joint[LFT_SDER*3+2] = xyz[2];  }
		if( idx == 6  ){ p_joint[LFT_EBOW*3+0] = xyz[0]; p_joint[LFT_EBOW*3+1] = xyz[1]; p_joint[LFT_EBOW*3+2] = xyz[2];  }
		if( idx == 7  ){ p_joint[LFT_HAND*3+0] = xyz[0]; p_joint[LFT_HAND*3+1] = xyz[1]; p_joint[LFT_HAND*3+2] = xyz[2];  }
		if( idx == 9  ){ p_joint[RHT_SDER*3+0] = xyz[0]; p_joint[RHT_SDER*3+1] = xyz[1]; p_joint[RHT_SDER*3+2] = xyz[2];  }
		if( idx == 10 ){ p_joint[RHT_EBOW*3+0] = xyz[0]; p_joint[RHT_EBOW*3+1] = xyz[1]; p_joint[RHT_EBOW*3+2] = xyz[2];  }
		if( idx == 11 ){ p_joint[RHT_HAND*3+0] = xyz[0]; p_joint[RHT_HAND*3+1] = xyz[1]; p_joint[RHT_HAND*3+2] = xyz[2];  }
		if( idx == 13 ){ p_joint[LFT_HIPS*3+0] = xyz[0]; p_joint[LFT_HIPS*3+1] = xyz[1]; p_joint[LFT_HIPS*3+2] = xyz[2];  }
		if( idx == 14 ){ p_joint[LFT_KNEE*3+0] = xyz[0]; p_joint[LFT_KNEE*3+1] = xyz[1]; p_joint[LFT_KNEE*3+2] = xyz[2];  }
		if( idx == 15 ){ p_joint[LFT_AKLE*3+0] = xyz[0]; p_joint[LFT_AKLE*3+1] = xyz[1]; p_joint[LFT_AKLE*3+2] = xyz[2];  }
		if( idx == 17 ){ p_joint[RHT_HIPS*3+0] = xyz[0]; p_joint[RHT_HIPS*3+1] = xyz[1]; p_joint[RHT_HIPS*3+2] = xyz[2];  }
		if( idx == 18 ){ p_joint[RHT_KNEE*3+0] = xyz[0]; p_joint[RHT_KNEE*3+1] = xyz[1]; p_joint[RHT_KNEE*3+2] = xyz[2];  }
		if( idx == 19 ){ p_joint[RHT_AKLE*3+0] = xyz[0]; p_joint[RHT_AKLE*3+1] = xyz[1]; p_joint[RHT_AKLE*3+2] = xyz[2];  }
	}
	for( int j = 0 ; j < JOINT_NUMBER ; j++ ){
		float pt[4];
		pt[0] = p_joint[j*3+0];
		pt[1] = p_joint[j*3+1];
		pt[2] = p_joint[j*3+2];
		pt[3] = 1;
		pdt2eval( extm, pt );
		p_joint[j*3 + 0] = pt[0];
		p_joint[j*3 + 1] = pt[1];
		p_joint[j*3 + 2] = pt[2];
		pixel2world( &p_joint[j*3] );
	}

	return load_successful;
}
void draw_depth_user( cv::Mat *image, float *depth ){

	float pmin,pmax;
	pmin = FLT_MAX;
	pmax = -FLT_MAX;
	//for( int p = 0 ; p < Q_HEIGHT*Q_WIDTH; p++ ){
	for( int y = 0 ; y < Q_HEIGHT ; y++ ){
	for( int x = 75 ; x < Q_WIDTH ; x++ ){
		int p = y*Q_WIDTH + x;
		if( depth[p] < BACKGROUND_DEPTH ){
			if( pmin > depth[p] ){
				pmin = depth[p];
			}
			if( pmax < depth[p] ){
				pmax = depth[p];
			}
		}
	}}

	//for( int p = 0 ; p < Q_HEIGHT*Q_WIDTH; p++ ){
	for( int y = 0 ; y < Q_HEIGHT ; y++ ){
	for( int x = 0 ; x < Q_WIDTH ; x++ ){
		int p = y*Q_WIDTH + x;
		if( depth[p] < BACKGROUND_DEPTH ){
			float gray = depth[p];
			gray = (175*(gray-pmin)/(pmax-pmin) + 50 );
			image->data[ p*3 + 0] = uchar( gray );
			image->data[ p*3 + 1] = uchar( gray );
			image->data[ p*3 + 2] = uchar( gray );
		}
		else{
			image->data[ p*3 + 0] = 0xff;
			image->data[ p*3 + 1] = 0xff;
			image->data[ p*3 + 2] = 0xff;
		}
	}}
}
void draw_joints( cv::Mat *image, float *joints ){
	int line_thickness;
	cv::Scalar middle, left, right;
	
	line_thickness     = 1;
	middle = cv::Scalar( 255,  0,255 );
	left   = cv::Scalar(   0,  0,255 );
	right  = cv::Scalar( 255,  0,  0 );

	float djoints[JOINT_NUMBER*3];
	for( int j = 0 ; j < JOINT_NUMBER ;j++ ){
		float pt[3];
		pt[0] = joints[j*3 + 0];
		pt[1] = joints[j*3 + 1];
		pt[2] = joints[j*3 + 2];
		world2pixel( pt );
		djoints[j*3+0] = pt[0];
		djoints[j*3+1] = pt[1];
		djoints[j*3+2] = pt[2];
	}
	rectangle(*image, Rect(djoints[ONE_HEAD*3+0]-2,djoints[ONE_HEAD*3+1]-2,4,4),CV_RGB(255,0,0),2);
	/*	
	cv::line( *image, cv::Point( int( djoints[ONE_HEAD*3+0] ), int( djoints[ONE_HEAD*3+1] )), cv::Point( int( djoints[ONE_SDER*3+0]), int( djoints[ONE_SDER*3+1] )), middle, line_thickness, 8, 0);
	
	//cv::line( *image, cv::Point( int( djoints[ONE_TRSO*3+0] ), int( djoints[ONE_TRSO*3+1] )), cv::Point( int( djoints[ONE_SDER*3+0]), int( djoints[ONE_SDER*3+1] )), middle, line_thickness, 8, 0);
	cv::line( *image, cv::Point( int( djoints[ONE_SDER*3+0] ), int( djoints[ONE_SDER*3+1] )), cv::Point( int( djoints[ONE_BLLY*3+0]), int( djoints[ONE_BLLY*3+1] )), middle, line_thickness, 8, 0);

	cv::line( *image, cv::Point( int( djoints[ONE_SDER*3+0] ), int( djoints[ONE_SDER*3+1] )), cv::Point( int( djoints[LFT_SDER*3+0]), int( djoints[LFT_SDER*3+1] )), left, line_thickness, 8, 0);
	cv::line( *image, cv::Point( int( djoints[LFT_EBOW*3+0] ), int( djoints[LFT_EBOW*3+1] )), cv::Point( int( djoints[LFT_SDER*3+0]), int( djoints[LFT_SDER*3+1] )), left, line_thickness, 8, 0);
	cv::line( *image, cv::Point( int( djoints[LFT_EBOW*3+0] ), int( djoints[LFT_EBOW*3+1] )), cv::Point( int( djoints[LFT_HAND*3+0]), int( djoints[LFT_HAND*3+1] )), left, line_thickness, 8, 0);
	cv::line( *image, cv::Point( int( djoints[ONE_BLLY*3+0] ), int( djoints[ONE_BLLY*3+1] )), cv::Point( int( djoints[LFT_HIPS*3+0]), int( djoints[LFT_HIPS*3+1] )), left, line_thickness, 8, 0);
	cv::line( *image, cv::Point( int( djoints[LFT_KNEE*3+0] ), int( djoints[LFT_KNEE*3+1] )), cv::Point( int( djoints[LFT_HIPS*3+0]), int( djoints[LFT_HIPS*3+1] )), left, line_thickness, 8, 0);
	cv::line( *image, cv::Point( int( djoints[LFT_KNEE*3+0] ), int( djoints[LFT_KNEE*3+1] )), cv::Point( int( djoints[LFT_AKLE*3+0]), int( djoints[LFT_AKLE*3+1] )), left, line_thickness, 8, 0);

	cv::line( *image, cv::Point( int( djoints[ONE_SDER*3+0] ), int( djoints[ONE_SDER*3+1] )), cv::Point( int( djoints[RHT_SDER*3+0]), int( djoints[RHT_SDER*3+1] )), right, line_thickness, 8, 0);
	cv::line( *image, cv::Point( int( djoints[RHT_EBOW*3+0] ), int( djoints[RHT_EBOW*3+1] )), cv::Point( int( djoints[RHT_SDER*3+0]), int( djoints[RHT_SDER*3+1] )), right, line_thickness, 8, 0);
	cv::line( *image, cv::Point( int( djoints[RHT_EBOW*3+0] ), int( djoints[RHT_EBOW*3+1] )), cv::Point( int( djoints[RHT_HAND*3+0]), int( djoints[RHT_HAND*3+1] )), right, line_thickness, 8, 0);
	cv::line( *image, cv::Point( int( djoints[ONE_BLLY*3+0] ), int( djoints[ONE_BLLY*3+1] )), cv::Point( int( djoints[RHT_HIPS*3+0]), int( djoints[RHT_HIPS*3+1] )), right, line_thickness, 8, 0);
	cv::line( *image, cv::Point( int( djoints[RHT_KNEE*3+0] ), int( djoints[RHT_KNEE*3+1] )), cv::Point( int( djoints[RHT_HIPS*3+0]), int( djoints[RHT_HIPS*3+1] )), right, line_thickness, 8, 0);
	cv::line( *image, cv::Point( int( djoints[RHT_KNEE*3+0] ), int( djoints[RHT_KNEE*3+1] )), cv::Point( int( djoints[RHT_AKLE*3+0]), int( djoints[RHT_AKLE*3+1] )), right, line_thickness, 8, 0);
	*/
}
void getPDTdepthpath( char *load_path, char *model, int fidx ){
	char suffix[128];
	load_path[0] = '\0';
	strcat( load_path, PDT_FOLDER );
	strcat( load_path, model );
	sprintf( suffix, "/parsed/parsed_depth_%i.txt", fidx );
	strcat( load_path, suffix );
}
void getPDTjointpath( char *load_path, char *model ){
	char suffix[128];
	load_path[0] = '\0';
	strcat( load_path, PDT_FOLDER );
	strcat( load_path, model );
	sprintf( suffix, "/joints.txt");
	strcat( load_path, suffix );
}
void get_save_path( char *save_path, char *folder, char *prefix, int idx ){
	char last_half[512];
	sprintf( last_half, "_%i.bin", idx );
	save_path[0] = '\0';
	strcat( save_path, folder );
	strcat( save_path, prefix );
	strcat( save_path, last_half  );	
}

void testPDT( char models[20][5] ){

	float extm[4][4] = {	-0.7515f,    0.0083f,   -0.6597f,   -0.0930f,
						 0.0257f,    0.9995f,   -0.0166f,   -1.0517f,
						 0.6592f,   -0.0294f,   -0.7514f,   -3.3886f,
						 0,          0,          0,		     1.0000f};
	float extm2[4][4] = {	-0.5856f,   0.0177f,	0.8104f,   -0.1058f,
							0.1047f,    0.9931f,	0.0541f,   -0.8531f,
							-0.8038f,   0.1165f,	-0.5834f,  -3.5738f,
							0,          0,          0,		    1.0000f};
	//m[0] = ;

	cv::Mat image( Q_HEIGHT, Q_WIDTH, CV_8UC3 );
	char load_path[1024];
	char joint_path[1024];
	int total_frames = 0;
	for( int m = 0; m < 20 ; m++ ){
		float ext_para[4][4];
		if( models[m][0] == 'F' && models[m][1] == '2' ){
			std::memcpy( ext_para, extm, sizeof(float)*16 );
		}
		else{
			std::memcpy( ext_para, extm2, sizeof(float)*16 );
		}
		getPDTjointpath( joint_path, models[m] );
		std::ifstream joint_in( joint_path);

		bool continue_loop = true;
		int fidx = 0;
		while( continue_loop ){
			getPDTdepthpath(load_path, models[m], fidx );

			int stats[6]; // model, action, action frame, total frame, good(1) or bad(0), closest(not used)
			float p_depth[Q_HEIGHT*Q_WIDTH];
			float p_user[ Q_HEIGHT*Q_WIDTH]; // 0 for background, 1 for foreground
			float p_joints[16*3];
			
			stats[0] = int(m/4);
			stats[1] = int(m%4);
			stats[2] = fidx;
			stats[3] = total_frames;
			stats[4] = 1; // good;
			stats[5] = 10000; // just a number

			if( load2joint( &joint_in, p_joints, ext_para ) ){
				float body_center[3];
				load2depth( p_depth, p_user, body_center, load_path, ext_para);
				p_joints[ONE_BLLY*3 + 0] = body_center[0];
				p_joints[ONE_BLLY*3 + 1] = body_center[1];
				p_joints[ONE_BLLY*3 + 2] = body_center[2];
				//printf( "(%s, %i, %i) ", models[m], m, fidx );
				printf( "MODEL%i ACTION%i ACTIONFRAME%i TOTALFRAME%i GOOD%i CLOSE%i\n", stats[0], stats[1], stats[2], stats[3], stats[4], stats[5]);
				draw_depth_user( &image, p_depth );
				draw_joints( &image, p_joints );
				cv::imshow( "pdt", image );
				cv::waitKey(1);

				char save_depth_path[1024];
				char save_user_path[ 1024];
				char save_joint_path[1024];
				char save_stat_path[ 1024];

				get_save_path( save_depth_path, SAVE_FOLDER, "pdt_time_depth", total_frames );
				get_save_path( save_user_path,  SAVE_FOLDER, "pdt_time_user",  total_frames );
				get_save_path( save_joint_path, SAVE_FOLDER, "pdt_time_joint", total_frames );
				get_save_path( save_stat_path,  SAVE_FOLDER, "pdt_time_stat",  total_frames );
				
				FILE *ofp;
				ofp = fopen( save_depth_path, "wb" );
				fwrite( &p_depth, sizeof(float), Q_WIDTH*Q_HEIGHT, ofp);
				fclose( ofp );
				ofp = fopen( save_user_path, "wb" );
				fwrite( &p_user, sizeof(float), Q_WIDTH*Q_HEIGHT, ofp);
				fclose( ofp );
				ofp = fopen( save_joint_path, "wb" );
				fwrite( &p_joints, sizeof(float), JOINT_NUMBER*3, ofp);
				fclose( ofp );
				ofp = fopen( save_stat_path, "wb" );
				fwrite( &stats, sizeof(int), 6, ofp);
				fclose( ofp );

				fidx += 1;
				total_frames += 1;
			}
			else{
				continue_loop = false;
			}
		}
		printf( "\n\n" );
		joint_in.close();
	}
	cv::waitKey();
}

int main()
{
	char models[20][5] = {	"F1D1", "F1D2", "F1D3", "F1D4",
							"F2D1", "F2D2", "F2D3", "F2D4",
							"M1D1", "M1D2", "M1D3", "M1D4",
							"M2D1", "M2D2", "M2D3", "M2D4",
							"M3D1", "M3D2", "M3D3", "M3D4" };

	testPDT( models );

	return 0;
}

