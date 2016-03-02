#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <time.h>
#include <iterator>
#include <omp.h>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp> 
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


//Should be set for each device and OS
#define WINDOW 0
#define TRAINING_DIR "/media/sda10/Study/Data/Human_pose_estimation/EVAL/indexed_ordered_by_diversity/"
#define SAVE_DIR "/media/sda10/Study/Data/Human_pose_estimation/Result_EVAL/"

#if WINDOW
#include <Windows.h>
#else
#include <boost/filesystem.hpp>
#endif



#define Q_WIDTH 320
#define Q_HEIGHT 240
#define BACKGROUND_DEPTH 100
#define COORDINATE_DIM 3
#define JOINT_NUMBER 16
#define CLUSTER_JOINT_NUMBER 15
#define RESULT_JOINT_NUMBER 12
#define SEARCH_RADIUS_XY 1.0f
#define SEARCH_RADIUS_Z 0.2f
#define SHOULDER_LENGTH 0.4f
#define PROB_START_TARGET_COMP 0.00f
#define PROB_CURRENT_COMP 0.20f

#define TEST 0
#define TRAINING 1
#define POINT_MATCHING 2

#define K_MEANS_CLUSTER_NUM 2
#define K_MEANS_ITER 10
#define K_MEANS_ATTEMPT 10
#define K_MEANS_EPS 0.001

#define NEAREST_NUM 2
#define FEATURE_ITER 1024
#define MIN_SAMPLE_NUM 1024
#define THR 0.1
#define RANDOM_SAMPLE 1000
#define NN_SEARCH_RADIUS 0.1f

#define TREE_NUM 3
#define BAGGING_RATIO 1

#define TRAIN_FRAME_NUM 5520
#define POINT_MATCHING_FRAME_NUM 1000
#define TEST_FRAME_NUM 2762	//model0 -> 2760, model1 -> 2762, model2 -> 2760
#define TRAIN_MODEL_1 0
#define TRAIN_MODEL_2 2
#define TEST_MODEL 1


#define HEAD 0
#define SDER 1
#define TRSO 2
#define BLLY 3
#define L_SDER 4
#define L_EBOW 5
#define L_HAND 6
#define L_HIPS 7
#define L_KNEE 8
#define L_ANKLE 9
#define R_SDER 10
#define R_EBOW 11
#define R_HAND 12
#define R_HIPS 13
#define R_KNEE 14
#define R_ANKLE 15
//ABOVE IS FOR PP_JOINT(FOR ALL JOINT)
//TRAIN EXCLUDE TRSO
//WHEN CLUSTERING(FINAL RESULT) -> TRSO, BLLY, L_HIP, R_HIP IS EXCLUDED



using namespace std;
using namespace cv;


void draw_depth_user(Mat *image, float *depth);
inline float rand_float(float min, float max);
inline void pixel2world(float pt[3]);
inline void world2pixel(float pt[3]);
vector<string> get_all_files_names_within_folder(string folder);

class vote
{
public:
	int x_pixel, y_pixel, priority;
	float depth;
	bool operator<(const vote& ref) const { return priority > ref.priority; }
};

class cluster
{
public:
	float **offset;
	float priority;
	bool operator<(const cluster& ref) const { return priority < ref.priority; }
};
class joint
{
public:
	int x_pixel, y_pixel;
	float x_world, y_world, depth;
};
class point
{
public:
	int x_pixel, y_pixel, frame_id;
	float x_world, y_world, depth;
	float **offset;

	void get_offset(joint* joint_train);
	bool split_left(point* point_train, float delta[5], float search_radius_xy, float search_radius_z);
};

class node
{
public:
	node(){ left_child = NULL; right_child = NULL; }

	int depth;
	float delta[5];
	float search_radius_xy, search_radius_z;
	vector<point*> sample;

	vector<cluster> clustered_offset;

	node* left_child;
	node* right_child;

};

void point::get_offset(joint* pp_joint)
{

	class tmp
	{
	public:
		tmp(float dist, float x, float y, float depth){ this->dist = dist; this->x = x; this->y = y; this->depth = depth; }
		float x, y, depth, dist;
		bool operator<(const tmp& ref) const { return dist < ref.dist; }
	};
	vector<tmp> min_dist_vec;
	min_dist_vec.reserve(JOINT_NUMBER);

	for (int joint_id = 0; joint_id < JOINT_NUMBER; joint_id++)
	{
		if (joint_id != TRSO)
		{
			float dist = sqrt(pow(x_world - pp_joint[joint_id].x_world, 2) + pow(y_world - pp_joint[joint_id].y_world, 2) + pow(depth - pp_joint[joint_id].depth, 2));
			tmp inst(dist, pp_joint[joint_id].x_world - x_world, pp_joint[joint_id].y_world - y_world, pp_joint[joint_id].depth - depth);	//offset = joint position vector - current position vector
			min_dist_vec.push_back(inst);
		}
	}
	sort(min_dist_vec.begin(), min_dist_vec.end());

	offset = new float*[NEAREST_NUM];
	for (int nid = 0; nid < NEAREST_NUM; nid++)
	{
		offset[nid] = new float[COORDINATE_DIM];
		offset[nid][0] = min_dist_vec[nid].x;
		offset[nid][1] = min_dist_vec[nid].y;
		offset[nid][2] = min_dist_vec[nid].depth;
	}
	
}
bool point::split_left(point* joint_train, float delta[5], float search_radius_xy, float search_radius_z)
{
	if (delta[0] <= 1.0f)
	{
		float pt1[3], pt2[3];
		pt1[0] = x_world + delta[0] * search_radius_xy;
		pt1[1] = y_world + delta[1] * search_radius_xy;
		pt1[2] = depth;
		world2pixel(pt1);
		pt2[0] = x_world + delta[2] * search_radius_xy;
		pt2[1] = y_world + delta[3] * search_radius_xy;
		pt2[2] = depth;
		world2pixel(pt2);

		int x1, y1, x2, y2;
		x1 = int(pt1[0] + 0.5f);
		y1 = int(pt1[1] + 0.5f);
		x2 = int(pt2[0] + 0.5f);
		y2 = int(pt2[1] + 0.5f);
		if (x1 < 0){ x1 = 0; }
		if (x2 < 0){ x2 = 0; }
		if (y1 < 0){ y1 = 0; }
		if (y2 < 0){ y2 = 0; }
		if (x1 >= Q_WIDTH){ x1 = Q_WIDTH - 1; }
		if (x2 >= Q_WIDTH){ x2 = Q_WIDTH - 1; }
		if (y1 >= Q_HEIGHT){ y1 = Q_HEIGHT - 1; }
		if (y2 >= Q_HEIGHT){ y2 = Q_HEIGHT - 1; }

		float v1, v2;
		v1 = joint_train[frame_id * Q_HEIGHT * Q_WIDTH + y1 * Q_WIDTH + x1].depth + delta[4] * search_radius_z;
		v2 = joint_train[frame_id * Q_HEIGHT * Q_WIDTH + y2 * Q_WIDTH + x2].depth;

		if (v1 < v2)
			return true;
		else
			return false;
	}

	if (delta[0] == 2.0f)
	{
		float pt2[3]; int x2, y2;

		pt2[0] = x_world + delta[2] * search_radius_xy;
		pt2[1] = y_world + delta[3] * search_radius_xy;
		pt2[2] = depth;
		world2pixel(pt2);

		x2 = int(pt2[0] + 0.5f);
		y2 = int(pt2[1] + 0.5f);
		if (x2 < 0){ x2 = 0; }
		if (y2 < 0){ y2 = 0; }
		if (x2 >= Q_WIDTH){ x2 = Q_WIDTH - 1; }
		if (y2 >= Q_HEIGHT){ y2 = Q_HEIGHT - 1; }

		float v1, v2;
		v1 = y_world + (delta[4] * search_radius_z);
		v2 = joint_train[frame_id * Q_HEIGHT * Q_WIDTH + y2 * Q_WIDTH + x2].depth;


		if (v1 < v2)
			return true;
		else
			return false;
	}

	if (delta[0] == 3.0f)
	{
		// search diameter is based for the offset instead of search search radius
		float diff = rand_float(-1, 1);
		if (delta[1] > delta[2] && delta[1] > delta[3]){
			diff = x_world - offset[0][0] + (delta[4] * search_radius_xy * 2);
		}
		if (delta[2] > delta[1] && delta[2] > delta[3]){
			diff = y_world - offset[0][1] + (delta[4] * search_radius_xy * 2);
		}
		if (delta[3] > delta[1] && delta[3] > delta[2]){
			diff = y_world - offset[0][1] + (delta[4] * search_radius_xy * 2);
		}
		if (diff > 0)
			return true;
		else
			return false;

	}
}

void split_node(node* tree_ptr, int depth, int& node_number, point* point_train, int& leaf_number, int tree_number)
{
	cout << tree_number << "th tree " << node_number << "th node is in training... ";
	cout << "sample: " << tree_ptr->sample.size() << endl;

	if (tree_ptr->sample.size() < MIN_SAMPLE_NUM)
	{
		leaf_number++;
		node_number++;
		//K-means clustering to offset
		Mat k_sample(Size(NEAREST_NUM * COORDINATE_DIM, tree_ptr->sample.size()), CV_32FC1);
		Mat best_label, center;
		for (int sid = 0; sid < tree_ptr->sample.size(); sid++)
		{
			
			for (int nid = 0; nid < NEAREST_NUM; nid++)
			{
				for (int cid = 0; cid < COORDINATE_DIM; cid++)
					k_sample.at<float>(sid, nid * COORDINATE_DIM + cid) = tree_ptr->sample[sid]->offset[nid][cid];
			}
		}

		kmeans(k_sample, K_MEANS_CLUSTER_NUM, best_label, TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, K_MEANS_ITER, K_MEANS_EPS), K_MEANS_ATTEMPT, KMEANS_PP_CENTERS, center);

		//figure2,3
		if (leaf_number == 10000 || leaf_number == 110000 || leaf_number == 210000 || leaf_number == 310000 || leaf_number == 410000 || leaf_number == 510000 || leaf_number == 610000 || leaf_number == 710000 || leaf_number == 810000 || leaf_number == 910000)
		{
			ofstream of;
			stringstream ss;
			ss << SAVE_DIR << "figure_2_3/" << leaf_number << "_world.txt";
			of.open(ss.str());
			ss.str("");
			ss.clear();

			for (int i = 0; i < tree_ptr->sample.size(); i++)
			{
				ss << tree_ptr->sample[i]->x_world << " " << tree_ptr->sample[i]->y_world << " " << tree_ptr->sample[i]->depth << " ";
				for (int oid = 0; oid < NEAREST_NUM; oid++)
				{
					for (int cid = 0; cid < COORDINATE_DIM; cid++)
						ss << tree_ptr->sample[i]->offset[oid][cid] << " ";
				}
				ss << tree_ptr->sample[i]->frame_id;
				ss << endl;
				of << ss.str();
				ss.str("");
				ss.clear();
			}
			of.close();

			ss << SAVE_DIR << "figure_2_3/" << leaf_number << "_cluster.txt";
			of.open(ss.str());
			ss.str("");
			ss.clear();

			for (int i = 0; i < center.rows; i++)
			{
				for (int j = 0; j < center.cols; j++)
					of << center.at<float>(i, j) << " ";
				of << endl;
			}
			of.close();



			vector<int> checked_fid;
			for (int sid = 0; sid < tree_ptr->sample.size(); sid++)
			{
				bool is_checked = false;
				for (int j = 0; j < checked_fid.size(); j++)
				{
					if (checked_fid[j] == tree_ptr->sample[sid]->frame_id)
						is_checked = true;
				}

				if (is_checked)
					continue;

				Mat display(Q_HEIGHT, Q_WIDTH, CV_8UC3);
				float* display_tmp = new float[Q_HEIGHT * Q_WIDTH];

				for (int i = 0; i < Q_HEIGHT * Q_WIDTH; i++)
					display_tmp[i] = point_train[tree_ptr->sample[sid]->frame_id * Q_HEIGHT * Q_WIDTH + i].depth;

				draw_depth_user(&display, display_tmp);

				checked_fid.push_back(tree_ptr->sample[sid]->frame_id);
				for (int ssid = 0; ssid < tree_ptr->sample.size(); ssid++)
				{
					if (tree_ptr->sample[sid]->frame_id == tree_ptr->sample[ssid]->frame_id)
					{
						float pt[3];
						pt[0] = tree_ptr->sample[ssid]->x_world;
						pt[1] = tree_ptr->sample[ssid]->y_world;
						pt[2] = tree_ptr->sample[ssid]->depth;
						world2pixel(pt);
						rectangle(display, Rect(pt[0] - 2, pt[1] - 2, 4, 4), CV_RGB(0, 255, 0), 2);	//original

						pt[0] = tree_ptr->sample[ssid]->x_world + k_sample.at<float>(ssid, 0);
						pt[1] = tree_ptr->sample[ssid]->y_world + k_sample.at<float>(ssid, 1);
						pt[2] = tree_ptr->sample[ssid]->depth + k_sample.at<float>(ssid, 2);
						world2pixel(pt);
						rectangle(display, Rect(pt[0] - 2, pt[1] - 2, 4, 4), CV_RGB(255, 0, 0), 2);	//off1

						pt[0] = tree_ptr->sample[ssid]->x_world + k_sample.at<float>(ssid, 3);
						pt[1] = tree_ptr->sample[ssid]->y_world + k_sample.at<float>(ssid, 4);
						pt[2] = tree_ptr->sample[ssid]->depth + k_sample.at<float>(ssid, 5);
						world2pixel(pt);
						rectangle(display, Rect(pt[0] - 2, pt[1] - 2, 4, 4), CV_RGB(0, 0, 255), 2);	//off2

					}


				}

				ss << SAVE_DIR << "figure_2_3/" << leaf_number << "_" << tree_ptr->sample[sid]->frame_id << ".jpg";
				imwrite(ss.str(), display);
				ss.str("");
				ss.clear();
				delete[] display_tmp;
				display.release();
			}
		}
		//figure_2_3

		tree_ptr->clustered_offset.reserve(K_MEANS_CLUSTER_NUM);
		for (int i = 0; i < K_MEANS_CLUSTER_NUM; i++)
		{
			cluster tmp;
			tmp.priority = 0;
			tmp.offset = new float*[NEAREST_NUM];
			for (int nid = 0; nid < NEAREST_NUM; nid++)
			{
				tmp.offset[nid] = new float[COORDINATE_DIM];
				for (int cid = 0; cid < COORDINATE_DIM; cid++)
					tmp.offset[nid][cid] = center.at<float>(i, nid * COORDINATE_DIM + cid);
			}
			tree_ptr->clustered_offset.push_back(tmp);
		}
		for (int i = 0; i < best_label.rows; i++)
			tree_ptr->clustered_offset[best_label.at<int>(i, 0)].priority++;
		for (int i = 0; i < tree_ptr->clustered_offset.size(); i++)
			tree_ptr->clustered_offset[i].priority /= (float)tree_ptr->sample.size();


		sort(tree_ptr->clustered_offset.begin(), tree_ptr->clustered_offset.end());

		return;
	}

	else
	{
		node_number++;
		tree_ptr->depth = depth;
		tree_ptr->left_child = new node();
		tree_ptr->right_child = new node();

		float save_min_energy = 0;

		float save_delta[5];
		float save_radius_xy, save_radius_z;



		#pragma omp parallel for num_threads(omp_get_max_threads())
		for (int feature_iter = 0; feature_iter < FEATURE_ITER; feature_iter++)
		{
			vector<point*> left_sample, right_sample;

			float delta[5], search_radius_xy, search_radius_z;


			delta[0] = rand_float(-1, 1);
			if (rand_float(0, 1) < PROB_START_TARGET_COMP + PROB_CURRENT_COMP)
			{
				delta[0] = 2;
				if (rand_float(0, PROB_START_TARGET_COMP + PROB_CURRENT_COMP) < PROB_START_TARGET_COMP)
				{
					delta[0] = 3;
				}
			}

			delta[1] = rand_float(-1.0, 1.0);
			delta[2] = rand_float(-1.0, 1.0);
			delta[3] = rand_float(-1.0, 1.0);
			delta[4] = rand_float(-1.0, 1.0);
			search_radius_xy = SHOULDER_LENGTH * rand_float(0.90f, 1.10f) * SEARCH_RADIUS_XY;
			search_radius_z = SHOULDER_LENGTH * rand_float(0.90f, 1.10f) * SEARCH_RADIUS_Z;


			float **offset_left_avg;
			float **offset_right_avg;


			offset_left_avg = new float*[NEAREST_NUM];
			offset_right_avg = new float*[NEAREST_NUM];
			for (int nid = 0; nid < NEAREST_NUM; nid++)
			{
				offset_left_avg[nid] = new float[COORDINATE_DIM];
				offset_right_avg[nid] = new float[COORDINATE_DIM];
			}

			for (int nid = 0; nid < NEAREST_NUM; nid++)
			{
				for (int cid = 0; cid < COORDINATE_DIM; cid++)
				{
					offset_left_avg[nid][cid] = 0;
					offset_right_avg[nid][cid] = 0;
				}
			}
		

			for (int sid = 0; sid < tree_ptr->sample.size(); sid++)
			{
				bool is_left = tree_ptr->sample[sid]->split_left(point_train, delta, search_radius_xy, search_radius_z);
				if (is_left)
				{
					left_sample.push_back(tree_ptr->sample[sid]);
					for (int nid = 0; nid < NEAREST_NUM; nid++)
					{
						for (int cid = 0; cid < COORDINATE_DIM; cid++)
							offset_left_avg[nid][cid] += tree_ptr->sample[sid]->offset[nid][cid];
					}
				}
				else
				{
					right_sample.push_back(tree_ptr->sample[sid]);
					for (int nid = 0; nid < NEAREST_NUM; nid++)
					{
						for (int cid = 0; cid < COORDINATE_DIM; cid++)
							offset_right_avg[nid][cid] += tree_ptr->sample[sid]->offset[nid][cid];
					}
				}
			}

			for (int nid = 0; nid < NEAREST_NUM; nid++)
			{
				for (int cid = 0; cid < COORDINATE_DIM; cid++)
				{
					offset_left_avg[nid][cid] /= (float)(left_sample.size());
					offset_right_avg[nid][cid] /= (float)(right_sample.size());
				}
			}


			float left_energy = 0;
			float right_energy = 0;
			for (int sid = 0; sid < left_sample.size(); sid++)
			{
				for (int nid = 0; nid < NEAREST_NUM; nid++)
				{
					for (int cid = 0; cid < COORDINATE_DIM; cid++)
						left_energy += pow(left_sample[sid]->offset[nid][cid] - offset_left_avg[nid][cid], 2);

				}
			}
			for (int sid = 0; sid < right_sample.size(); sid++)
			{
				for (int nid = 0; nid < NEAREST_NUM; nid++)
				{
					for (int cid = 0; cid < COORDINATE_DIM; cid++)
						right_energy += pow(right_sample[sid]->offset[nid][cid] - offset_right_avg[nid][cid], 2);

				}
			}


			left_energy *= ((float)left_sample.size() / (float)tree_ptr->sample.size());
			right_energy *= ((float)right_sample.size() / (float)tree_ptr->sample.size());

			if (feature_iter == 0)
				save_min_energy = left_energy + right_energy + 1;

			if (left_energy + right_energy < save_min_energy)
			{
				save_min_energy = left_energy + right_energy;
				save_delta[0] = delta[0];
				save_delta[1] = delta[1];
				save_delta[2] = delta[2];
				save_delta[3] = delta[3];
				save_delta[4] = delta[4];
				save_radius_xy = search_radius_xy;
				save_radius_z = search_radius_z;
			}

			for (int nid = 0; nid < NEAREST_NUM; nid++)
			{
				delete[] offset_left_avg[nid];
				delete[] offset_right_avg[nid];
			}
			delete[] offset_left_avg;
			delete[] offset_right_avg;
		}

		tree_ptr->left_child->depth = depth + 1;
		tree_ptr->right_child->depth = depth + 1;
		tree_ptr->delta[0] = save_delta[0];
		tree_ptr->delta[1] = save_delta[1];
		tree_ptr->delta[2] = save_delta[2];
		tree_ptr->delta[3] = save_delta[3];
		tree_ptr->delta[4] = save_delta[4];
		tree_ptr->search_radius_xy = save_radius_xy;
		tree_ptr->search_radius_z = save_radius_z;

		for (int sid = 0; sid < tree_ptr->sample.size(); sid++)
		{
			bool is_left = tree_ptr->sample[sid]->split_left(point_train, tree_ptr->delta, tree_ptr->search_radius_xy, tree_ptr->search_radius_z);
			if (is_left)
				tree_ptr->left_child->sample.push_back(tree_ptr->sample[sid]);
			else
				tree_ptr->right_child->sample.push_back(tree_ptr->sample[sid]);
		}


		if (tree_ptr->left_child->sample.size() < MIN_SAMPLE_NUM || tree_ptr->right_child->sample.size() < MIN_SAMPLE_NUM)
		{
			delete tree_ptr->left_child;
			delete tree_ptr->right_child;
			tree_ptr->left_child = NULL;
			tree_ptr->right_child = NULL;
			leaf_number++;


			//K-means clustering to offset
			Mat k_sample(Size(NEAREST_NUM * COORDINATE_DIM, tree_ptr->sample.size()), CV_32FC1);
			Mat best_label, center;
			for (int sid = 0; sid < tree_ptr->sample.size(); sid++)
			{

				for (int nid = 0; nid < NEAREST_NUM; nid++)
				{
					for (int cid = 0; cid < COORDINATE_DIM; cid++)
						k_sample.at<float>(sid, nid * COORDINATE_DIM + cid) = tree_ptr->sample[sid]->offset[nid][cid];
				}
			}

			kmeans(k_sample, K_MEANS_CLUSTER_NUM, best_label, TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, K_MEANS_ITER, K_MEANS_EPS), K_MEANS_ATTEMPT, KMEANS_PP_CENTERS, center);

			//figure_2_3
			if (leaf_number == 10000 || leaf_number == 110000 || leaf_number == 210000 || leaf_number == 310000 || leaf_number == 410000 || leaf_number == 510000 || leaf_number == 610000 || leaf_number == 710000 || leaf_number == 810000 || leaf_number == 910000)
			{
				ofstream of;
				stringstream ss;
				ss << SAVE_DIR << "figure_2_3/" << leaf_number << "_world.txt";
				of.open(ss.str());
				ss.str("");
				ss.clear();

				for (int i = 0; i < tree_ptr->sample.size(); i++)
				{
					ss << tree_ptr->sample[i]->x_world << " " << tree_ptr->sample[i]->y_world << " " << tree_ptr->sample[i]->depth << " ";
					for (int oid = 0; oid < NEAREST_NUM; oid++)
					{
						for (int cid = 0; cid < COORDINATE_DIM; cid++)
							ss << tree_ptr->sample[i]->offset[oid][cid] << " ";
					}
					ss << tree_ptr->sample[i]->frame_id;
					ss << endl;
					of << ss.str();
					ss.str("");
					ss.clear();
				}
				of.close();

				ss << SAVE_DIR << "figure_2_3/" << leaf_number << "_cluster.txt";
				of.open(ss.str());
				ss.str("");
				ss.clear();

				for (int i = 0; i < center.rows; i++)
				{
					for (int j = 0; j < center.cols; j++)
						of << center.at<float>(i, j) << " ";
					of << endl;
				}
				of.close();



				vector<int> checked_fid;
				for (int sid = 0; sid < tree_ptr->sample.size(); sid++)
				{
					bool is_checked = false;
					for (int j = 0; j < checked_fid.size(); j++)
					{
						if (checked_fid[j] == tree_ptr->sample[sid]->frame_id)
							is_checked = true;
					}

					if (is_checked)
						continue;

					Mat display(Q_HEIGHT, Q_WIDTH, CV_8UC3);
					float* display_tmp = new float[Q_HEIGHT * Q_WIDTH];

					for (int i = 0; i < Q_HEIGHT * Q_WIDTH; i++)
						display_tmp[i] = point_train[tree_ptr->sample[sid]->frame_id * Q_HEIGHT * Q_WIDTH + i].depth;

					draw_depth_user(&display, display_tmp);

					checked_fid.push_back(tree_ptr->sample[sid]->frame_id);
					for (int ssid = 0; ssid < tree_ptr->sample.size(); ssid++)
					{
						if (tree_ptr->sample[sid]->frame_id == tree_ptr->sample[ssid]->frame_id)
						{
							float pt[3];
							pt[0] = tree_ptr->sample[ssid]->x_world;
							pt[1] = tree_ptr->sample[ssid]->y_world;
							pt[2] = tree_ptr->sample[ssid]->depth;
							world2pixel(pt);
							rectangle(display, Rect(pt[0] - 2, pt[1] - 2, 4, 4), CV_RGB(0, 255, 0), 2);	//original

							pt[0] = tree_ptr->sample[ssid]->x_world + k_sample.at<float>(ssid, 0);
							pt[1] = tree_ptr->sample[ssid]->y_world + k_sample.at<float>(ssid, 1);
							pt[2] = tree_ptr->sample[ssid]->depth + k_sample.at<float>(ssid, 2);
							world2pixel(pt);
							rectangle(display, Rect(pt[0] - 2, pt[1] - 2, 4, 4), CV_RGB(255, 0, 0), 2);	//off1

							pt[0] = tree_ptr->sample[ssid]->x_world + k_sample.at<float>(ssid, 3);
							pt[1] = tree_ptr->sample[ssid]->y_world + k_sample.at<float>(ssid, 4);
							pt[2] = tree_ptr->sample[ssid]->depth + k_sample.at<float>(ssid, 5);
							world2pixel(pt);
							rectangle(display, Rect(pt[0] - 2, pt[1] - 2, 4, 4), CV_RGB(0, 0, 255), 2);	//off2

						}


					}

					ss << SAVE_DIR << "figure_2_3/" << leaf_number << "_" << tree_ptr->sample[sid]->frame_id << ".jpg";
					imwrite(ss.str(), display);
					ss.str("");
					ss.clear();
					delete[] display_tmp;
					display.release();
				}
			}
			//figure_2_3


			tree_ptr->clustered_offset.reserve(K_MEANS_CLUSTER_NUM);
			for (int i = 0; i < K_MEANS_CLUSTER_NUM; i++)
			{
				cluster tmp;
				tmp.priority = 0;
				tmp.offset = new float*[NEAREST_NUM];
				for (int nid = 0; nid < NEAREST_NUM; nid++)
				{
					tmp.offset[nid] = new float[COORDINATE_DIM];
					for (int cid = 0; cid < COORDINATE_DIM; cid++)
						tmp.offset[nid][cid] = center.at<float>(i, nid * COORDINATE_DIM + cid);
				}
				tree_ptr->clustered_offset.push_back(tmp);
			}
			for (int i = 0; i < best_label.rows; i++)
				tree_ptr->clustered_offset[best_label.at<int>(i, 0)].priority++;
			for (int i = 0; i < tree_ptr->clustered_offset.size(); i++)
				tree_ptr->clustered_offset[i].priority /= (float)tree_ptr->sample.size();


			sort(tree_ptr->clustered_offset.begin(), tree_ptr->clustered_offset.end());

			return;
		}


		split_node(tree_ptr->left_child, depth + 1, node_number, point_train, leaf_number, tree_number);
		split_node(tree_ptr->right_child, depth + 1, node_number, point_train, leaf_number, tree_number);

	}
}

void tree_train(point* point_train, joint** joint_train, vector<node*>& tree_head)
{
	node* tree_ptr;
	
	//find nearest joints and extreact feature for all pixels
	cout << "finding " << NEAREST_NUM << "-nearest joint for all pixels" << endl;
	for (int i = 0; i < TRAIN_FRAME_NUM * Q_HEIGHT * Q_WIDTH; i++)
	{

		int frame_id = i / (Q_HEIGHT * Q_WIDTH);

		if (point_train[i].depth != BACKGROUND_DEPTH)
			point_train[i].get_offset(joint_train[frame_id]);
		
	}

	for (int tid = 0; tid < TREE_NUM; tid++)
	{
		tree_ptr = new node();

		if (TREE_NUM == 1)
		{
			for (int i = 0; i < TRAIN_FRAME_NUM * Q_HEIGHT * Q_WIDTH; i++)
			{
				if (point_train[i].depth != BACKGROUND_DEPTH)
					tree_ptr->sample.push_back(point_train + i);
			}
		}
		else
		{
			vector<int> sampled_fid;
			int sampled_frame = 0;
			for (int iter = 0;; iter++)
			{
				int fid = rand() % TRAIN_FRAME_NUM;
				bool is_duplacated = false;
				for (int sfid = 0; sfid < sampled_fid.size(); sfid++)
				{
					if (fid == sampled_fid[sfid])
					{
						is_duplacated = true;
						break;
					}
				}
				
				if (is_duplacated)
					continue;

				sampled_fid.push_back(fid);

				for (int i = 0; i < Q_WIDTH * Q_HEIGHT; i++)
				{
					if (point_train[fid * Q_HEIGHT * Q_WIDTH + i].depth != BACKGROUND_DEPTH)
						tree_ptr->sample.push_back(point_train + fid * Q_HEIGHT * Q_WIDTH + i);
				}
				sampled_frame++;

				if (sampled_frame == BAGGING_RATIO * TRAIN_FRAME_NUM)
					break;
			}
		}

		int node_number = 0;
		int leaf_number = 0;
		split_node(tree_ptr, 0, node_number, point_train, leaf_number, tree_head.size());
		tree_head.push_back(tree_ptr);
	}
}
void tree_traversal(point* point_test, joint** joint_test, float** center_train, joint** joint_train, vector<node*> tree_head)
{

	for (int test_tree_number = 1; test_tree_number <= tree_head.size(); test_tree_number+=1)
	{
		node* tree_ptr;
		stringstream ss;

		ofstream average_result;
		ss << SAVE_DIR << "average_result_tree_num_" << test_tree_number << "_.txt";
		average_result.open(ss.str());
		ss.str("");
		ss.clear();
		int average_result_count[RESULT_JOINT_NUMBER];
		for (int i = 0; i < RESULT_JOINT_NUMBER; i++)
			average_result_count[i] = 0;

		ofstream upper_bound_result;
		ss << SAVE_DIR << "upper_bound_result_tree_num_" << test_tree_number << "_.txt";
		upper_bound_result.open(ss.str());
		ss.str("");
		ss.clear();
		int upper_bound_result_count[RESULT_JOINT_NUMBER];
		for (int i = 0; i < RESULT_JOINT_NUMBER; i++)
			upper_bound_result_count[i] = 0;

		ofstream time_result;
		ss << SAVE_DIR << "time_tree_num_" << test_tree_number << "_.txt";
		time_result.open(ss.str());
		ss.str("");
		ss.clear();

		clock_t tot_frame_begin, tot_frame_end;
		tot_frame_begin = clock();
		for (int f = 0; f < TEST_FRAME_NUM; f++)
		{
			cout << test_tree_number << " th tree " << f << "/" << TEST_FRAME_NUM << " th frame is in test..." << endl;

			time_result << "FID: " << f << " ";

			clock_t tot_begin, tot_end;
			tot_begin = clock();

			float pt_number = 0;
			float center_pt[3];
			center_pt[0] = center_pt[1] = center_pt[2] = 0;

			Mat display(Q_HEIGHT, Q_WIDTH, CV_8UC3);
			float* display_tmp = new float[Q_HEIGHT * Q_WIDTH];

			for (int i = 0; i < Q_HEIGHT * Q_WIDTH; i++)
			{
				display_tmp[i] = point_test[f * Q_HEIGHT * Q_WIDTH + i].depth;

				if (point_test[f * Q_HEIGHT * Q_WIDTH + i].depth != BACKGROUND_DEPTH)
				{
					center_pt[0] += point_test[f * Q_HEIGHT * Q_WIDTH + i].x_world;
					center_pt[1] += point_test[f * Q_HEIGHT * Q_WIDTH + i].y_world;
					center_pt[2] += point_test[f * Q_HEIGHT * Q_WIDTH + i].depth;
					pt_number++;
				}
			}
			center_pt[0] /= pt_number;
			center_pt[1] /= pt_number;
			center_pt[2] /= pt_number;


			draw_depth_user(&display, display_tmp);
			delete[] display_tmp;


			Mat sample;
			vector<float*> clustered_points;
			//voting by traversal of tree node
			clock_t vote_begin, vote_end;
			vote_begin = clock();
			for (int i = 0; i < Q_HEIGHT * Q_WIDTH; i++)
			{

				if (point_test[f * Q_HEIGHT * Q_WIDTH + i].depth == BACKGROUND_DEPTH)
					continue;

				float **average_offset;
				average_offset = new float*[NEAREST_NUM];
				for (int nid = 0; nid < NEAREST_NUM; nid++)
				{
					average_offset[nid] = new float[COORDINATE_DIM];
					for (int cid = 0; cid < COORDINATE_DIM; cid++)
						average_offset[nid][cid] = 0;
				}
				

				for (int tid = 0; tid < test_tree_number; tid++)
				{

					tree_ptr = tree_head[tid];

					float **estimated_offset;
					estimated_offset = new float*[NEAREST_NUM];
					for (int nid = 0; nid < NEAREST_NUM; nid++)
						estimated_offset[nid] = new float[COORDINATE_DIM];

					for (;;)
					{
						if (tree_ptr->left_child == NULL && tree_ptr->right_child == NULL)
						{
							//get a offset proportional to weight			
							float priority = (float)(rand() % 100) / (float)100;
							float prob_accul = 0;
							for (int cluster_id = 0; cluster_id < tree_ptr->clustered_offset.size(); cluster_id++)
							{
								prob_accul += tree_ptr->clustered_offset[cluster_id].priority;
								if (priority <= prob_accul)
								{
									for (int nid = 0; nid < NEAREST_NUM; nid++)
									{
										for (int cid = 0; cid < COORDINATE_DIM; cid++)
											estimated_offset[nid][cid] = tree_ptr->clustered_offset[cluster_id].offset[nid][cid];
									}
									break;
								}
							}
							break;

						}
						else
						{
							bool is_left = point_test[f * Q_HEIGHT * Q_WIDTH + i].split_left(point_test, tree_ptr->delta, tree_ptr->search_radius_xy, tree_ptr->search_radius_z);

							if (is_left)
								tree_ptr = tree_ptr->left_child;
							else
								tree_ptr = tree_ptr->right_child;
						}
					}

					
					
					for (int nid = 0; nid < NEAREST_NUM; nid++)
					{
						average_offset[nid][0] += point_test[f * Q_HEIGHT * Q_WIDTH + i].x_world + estimated_offset[nid][0];
						average_offset[nid][1] += point_test[f * Q_HEIGHT * Q_WIDTH + i].y_world + estimated_offset[nid][1];
						average_offset[nid][2] += point_test[f * Q_HEIGHT * Q_WIDTH + i].depth + estimated_offset[nid][2];
					}

					for (int nid = 0; nid < NEAREST_NUM; nid++)
						delete[] estimated_offset[nid];
					delete[] estimated_offset;
					
				}

				for (int nid = 0; nid < NEAREST_NUM; nid++)
				{
					average_offset[nid][0] /= (float)test_tree_number;
					average_offset[nid][1] /= (float)test_tree_number;
					average_offset[nid][2] /= (float)test_tree_number;
				}

				
				Mat estimated_sample(Size(COORDINATE_DIM, 1), CV_32FC1);
				for (int nid = 0; nid < NEAREST_NUM; nid++)
				{
					estimated_sample.at<float>(0, 0) = average_offset[nid][0];
					estimated_sample.at<float>(0, 1) = average_offset[nid][1];
					estimated_sample.at<float>(0, 2) = average_offset[nid][2];

					sample.push_back(estimated_sample.clone());
				}

				for (int nid = 0; nid < NEAREST_NUM; nid++)
					delete[] average_offset[nid];
				delete[] average_offset;
				
			}
			vote_end = clock();
			time_result << "VOTING: " << (double)(vote_end - vote_begin) / CLOCKS_PER_SEC << " ";



			//K-means for clustering
			clock_t cluster_begin, cluster_end;
			cluster_begin = clock();

			Mat random_sample;
			for (int iter = 0; iter < RANDOM_SAMPLE; iter++)
			{
				int sid = rand_float(0, sample.rows - 1);
				random_sample.push_back(sample.row(sid));
			}


			Mat best_label, center;
			kmeans(random_sample, CLUSTER_JOINT_NUMBER, best_label, TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, K_MEANS_ITER, K_MEANS_EPS), K_MEANS_ATTEMPT, KMEANS_PP_CENTERS, center);
			for (int i = 0; i < CLUSTER_JOINT_NUMBER; i++)
			{
				float *pt = new float[COORDINATE_DIM];
				pt[0] = center.at<float>(i, 0);
				pt[1] = center.at<float>(i, 1);
				pt[2] = center.at<float>(i, 2);

				clustered_points.push_back(pt);
				
			}

			for (int cid = 0; cid < CLUSTER_JOINT_NUMBER; cid++)
			{
				float count = 0;
				float pt[3];
				pt[0] = pt[1] = pt[2] = 0;
				for (int pid = 0; pid < sample.rows; pid++)
				{
					float dist = sqrt(pow(clustered_points[cid][0] - sample.at<float>(pid, 0), 2) + pow(clustered_points[cid][1] - sample.at<float>(pid, 1), 2) + pow(clustered_points[cid][2] - sample.at<float>(pid, 2), 2));
					if (dist <= NN_SEARCH_RADIUS)
					{
						count++;
						pt[0] += sample.at<float>(pid, 0);
						pt[1] += sample.at<float>(pid, 1);
						pt[2] += sample.at<float>(pid, 2);
					}

				}

				if (count == 0)	//outlier
					continue;


				pt[0] /= count;
				pt[1] /= count;
				pt[2] /= count;

				clustered_points[cid][0] = pt[0];
				clustered_points[cid][1] = pt[1];
				clustered_points[cid][2] = pt[2];

				world2pixel(pt);
				//rectangle(display, Rect(pt[0] - 2, pt[1] - 2, 4, 4), CV_RGB(255, 0, 0), 2);

			}
			cluster_end = clock();
			time_result << "CLUSTERING: " << (double)(cluster_end - cluster_begin) / CLOCKS_PER_SEC << " ";

			//labeling using point matching
			clock_t label_begin, label_end;
			label_begin = clock();
			float min_dist_sum = 9999;
			vector<int> save_cluster_id;
			for (int train_f = 0; train_f < POINT_MATCHING_FRAME_NUM; train_f++)
			{
				float template_center[3];
				template_center[0] = center_train[train_f][0];
				template_center[1] = center_train[train_f][1];
				template_center[2] = center_train[train_f][2];

				float align[3];

				align[0] = center_pt[0] - template_center[0];
				align[1] = center_pt[1] - template_center[1];
				align[2] = center_pt[2] - template_center[2];

				float dist_sum = 0;
				vector<int> save_cid_tmp;
				for (int jid = 0; jid < JOINT_NUMBER; jid++)
				{

					if (jid == TRSO)
						continue;

					float min_dist = 9999;
					int save_cid;
					for (int cid = 0; cid < CLUSTER_JOINT_NUMBER; cid++)
					{
						float dist = sqrt(pow(joint_train[train_f][jid].x_world - (clustered_points[cid][0] - align[0]), 2) + pow(joint_train[train_f][jid].y_world - (clustered_points[cid][1] - align[1]), 2) + pow(joint_train[train_f][jid].depth - (clustered_points[cid][2] - align[2]), 2));

						if (min_dist > dist)
						{
							min_dist = dist;
							save_cid = cid;
						}
					}

					save_cid_tmp.push_back(save_cid);
					dist_sum += min_dist;
				}

				if (min_dist_sum > dist_sum)
				{
					min_dist_sum = dist_sum;
					save_cluster_id = save_cid_tmp;
				}
			}
			label_end = clock();
			time_result << "LABELING: " << (double)(label_end - label_begin) / CLOCKS_PER_SEC << " ";


			////for figure1, figure4
			//ofstream draw_3d;
			//ss << SAVE_DIR << f << "_depth.txt";
			//draw_3d.open(ss.str());
			//ss.str("");
			//ss.clear();

			//for (int i = 0; i < Q_HEIGHT * Q_WIDTH; i++)
			//{
			//	if (point_test[f * Q_HEIGHT * Q_WIDTH + i].depth != BACKGROUND_DEPTH)
			//	{
			//		ss << point_test[f * Q_HEIGHT * Q_WIDTH + i].x_world << " " << point_test[f * Q_HEIGHT * Q_WIDTH + i].y_world << " " << point_test[f * Q_HEIGHT * Q_WIDTH + i].depth << endl;
			//		draw_3d << ss.str();
			//		ss.str("");
			//		ss.clear();
			//	}
			//}
			//draw_3d.close();

			//ss << SAVE_DIR << f << "_estimated.txt";
			//draw_3d.open(ss.str());
			//ss.str("");
			//ss.clear();

			//for (int cid = 0; cid < clustered_points.size(); cid++)
			//{
			//	ss << clustered_points[cid][0] << " " << clustered_points[cid][1] << " " << clustered_points[cid][2] << endl;
			//	draw_3d << ss.str();
			//	ss.str("");
			//	ss.clear();
			//}
			//draw_3d.close();

			//ss << SAVE_DIR << f << "_sample.txt";
			//draw_3d.open(ss.str());
			//ss.str("");
			//ss.clear();

			//for (int sid = 0; sid < sample.rows; sid++)
			//{
			//	ss << sample.at<float>(sid, 0) << " " << sample.at<float>(sid, 1) << " " << sample.at<float>(sid, 2) << endl;
			//	draw_3d << ss.str();
			//	ss.str("");
			//	ss.clear();
			//}
			//draw_3d.close();



			//ss << SAVE_DIR << f << "_gt.txt";
			//draw_3d.open(ss.str());
			//ss.str("");
			//ss.clear();

			//for (int joint_index = 0; joint_index < JOINT_NUMBER; joint_index++)
			//{
			//	if (joint_index != TRSO)
			//	{
			//		ss << joint_test[f][joint_index].x_world << " " << joint_test[f][joint_index].y_world << " " << joint_test[f][joint_index].depth << endl;
			//		draw_3d << ss.str();
			//		ss.str("");
			//		ss.clear();
			//	}
			//	
			//}
			//draw_3d.close();



			//mAP mesure for each joint
			for (int jid = 0; jid < JOINT_NUMBER; jid++)
			{
				stringstream ss;
				int cid;
				int r, g, b;

				if (jid == TRSO)
					continue;

				if (jid == HEAD)
				{

					cid = save_cluster_id[0];
					float dist = sqrt(pow(joint_test[f][jid].x_world - clustered_points[cid][0], 2) + pow(joint_test[f][jid].y_world - clustered_points[cid][1], 2) + pow(joint_test[f][jid].depth - clustered_points[cid][2], 2));
					if (dist <= THR)
						average_result_count[0]++;
					r = 102;
					g = 102;
					b = 0;	//khaki

					float pt[3];
					pt[0] = clustered_points[cid][0];
					pt[1] = clustered_points[cid][1];
					pt[2] = clustered_points[cid][2];
					world2pixel(pt);


					rectangle(display, Rect(pt[0] - 2, pt[1] - 2, 4, 4), CV_RGB(r, g, b), 2);

				}
				else if (jid == SDER)
				{

					cid = save_cluster_id[1];
					float dist = sqrt(pow(joint_test[f][jid].x_world - clustered_points[cid][0], 2) + pow(joint_test[f][jid].y_world - clustered_points[cid][1], 2) + pow(joint_test[f][jid].depth - clustered_points[cid][2], 2));
					if (dist <= THR)
						average_result_count[1]++;
					r = 51;
					g = 51;
					b = 255;	//blue

					float pt[3];
					pt[0] = clustered_points[cid][0];
					pt[1] = clustered_points[cid][1];
					pt[2] = clustered_points[cid][2];
					world2pixel(pt);


					rectangle(display, Rect(pt[0] - 2, pt[1] - 2, 4, 4), CV_RGB(r, g, b), 2);

				}
				else if (jid == L_SDER)
				{

					cid = save_cluster_id[3];
					float dist = sqrt(pow(joint_test[f][jid].x_world - clustered_points[cid][0], 2) + pow(joint_test[f][jid].y_world - clustered_points[cid][1], 2) + pow(joint_test[f][jid].depth - clustered_points[cid][2], 2));
					if (dist <= THR)
						average_result_count[2]++;
					r = 51;
					g = 204;
					b = 51;	//green

					float pt[3];
					pt[0] = clustered_points[cid][0];
					pt[1] = clustered_points[cid][1];
					pt[2] = clustered_points[cid][2];
					world2pixel(pt);


					rectangle(display, Rect(pt[0] - 2, pt[1] - 2, 4, 4), CV_RGB(r, g, b), 2);

				}
				else if (jid == L_EBOW)
				{

					cid = save_cluster_id[4];
					float dist = sqrt(pow(joint_test[f][jid].x_world - clustered_points[cid][0], 2) + pow(joint_test[f][jid].y_world - clustered_points[cid][1], 2) + pow(joint_test[f][jid].depth - clustered_points[cid][2], 2));
					if (dist <= THR)
						average_result_count[3]++;
					r = 204;
					g = 51;
					b = 204;	//purple

					float pt[3];
					pt[0] = clustered_points[cid][0];
					pt[1] = clustered_points[cid][1];
					pt[2] = clustered_points[cid][2];
					world2pixel(pt);


					rectangle(display, Rect(pt[0] - 2, pt[1] - 2, 4, 4), CV_RGB(r, g, b), 2);

				}
				else if (jid == L_HAND)
				{

					cid = save_cluster_id[5];
					float dist = sqrt(pow(joint_test[f][jid].x_world - clustered_points[cid][0], 2) + pow(joint_test[f][jid].y_world - clustered_points[cid][1], 2) + pow(joint_test[f][jid].depth - clustered_points[cid][2], 2));
					if (dist <= THR)
						average_result_count[4]++;
					r = 51;
					g = 255;
					b = 255;	//sky

					float pt[3];
					pt[0] = clustered_points[cid][0];
					pt[1] = clustered_points[cid][1];
					pt[2] = clustered_points[cid][2];
					world2pixel(pt);


					rectangle(display, Rect(pt[0] - 2, pt[1] - 2, 4, 4), CV_RGB(r, g, b), 2);

				}
				else if (jid == L_KNEE)
				{

					cid = save_cluster_id[7];
					float dist = sqrt(pow(joint_test[f][jid].x_world - clustered_points[cid][0], 2) + pow(joint_test[f][jid].y_world - clustered_points[cid][1], 2) + pow(joint_test[f][jid].depth - clustered_points[cid][2], 2));
					if (dist <= THR)
						average_result_count[5]++;
					r = 204;
					g = 102;
					b = 0;	//orange

					float pt[3];
					pt[0] = clustered_points[cid][0];
					pt[1] = clustered_points[cid][1];
					pt[2] = clustered_points[cid][2];
					world2pixel(pt);


					rectangle(display, Rect(pt[0] - 2, pt[1] - 2, 4, 4), CV_RGB(r, g, b), 2);

				}
				else if (jid == L_ANKLE)
				{

					cid = save_cluster_id[8];
					float dist = sqrt(pow(joint_test[f][jid].x_world - clustered_points[cid][0], 2) + pow(joint_test[f][jid].y_world - clustered_points[cid][1], 2) + pow(joint_test[f][jid].depth - clustered_points[cid][2], 2));
					if (dist <= THR)
						average_result_count[6]++;
					r = 255;
					g = 255;
					b = 0;	//yellow

					float pt[3];
					pt[0] = clustered_points[cid][0];
					pt[1] = clustered_points[cid][1];
					pt[2] = clustered_points[cid][2];
					world2pixel(pt);


					rectangle(display, Rect(pt[0] - 2, pt[1] - 2, 4, 4), CV_RGB(r, g, b), 2);

				}
				else if (jid == R_SDER)
				{

					cid = save_cluster_id[9];
					float dist = sqrt(pow(joint_test[f][jid].x_world - clustered_points[cid][0], 2) + pow(joint_test[f][jid].y_world - clustered_points[cid][1], 2) + pow(joint_test[f][jid].depth - clustered_points[cid][2], 2));
					if (dist <= THR)
						average_result_count[7]++;
					r = 102;
					g = 0;
					b = 51;	//dark purple

					float pt[3];
					pt[0] = clustered_points[cid][0];
					pt[1] = clustered_points[cid][1];
					pt[2] = clustered_points[cid][2];
					world2pixel(pt);


					rectangle(display, Rect(pt[0] - 2, pt[1] - 2, 4, 4), CV_RGB(r, g, b), 2);

				}
				else if (jid == R_EBOW)
				{

					cid = save_cluster_id[10];
					float dist = sqrt(pow(joint_test[f][jid].x_world - clustered_points[cid][0], 2) + pow(joint_test[f][jid].y_world - clustered_points[cid][1], 2) + pow(joint_test[f][jid].depth - clustered_points[cid][2], 2));
					if (dist <= THR)
						average_result_count[8]++;
					r = 102;
					g = 51;
					b = 0;	//brown

					float pt[3];
					pt[0] = clustered_points[cid][0];
					pt[1] = clustered_points[cid][1];
					pt[2] = clustered_points[cid][2];
					world2pixel(pt);


					rectangle(display, Rect(pt[0] - 2, pt[1] - 2, 4, 4), CV_RGB(r, g, b), 2);

				}
				else if (jid == R_HAND)
				{

					cid = save_cluster_id[11];
					float dist = sqrt(pow(joint_test[f][jid].x_world - clustered_points[cid][0], 2) + pow(joint_test[f][jid].y_world - clustered_points[cid][1], 2) + pow(joint_test[f][jid].depth - clustered_points[cid][2], 2));
					if (dist <= THR)
						average_result_count[9]++;
					r = 0;
					g = 153;
					b = 153;	//dark sky

					float pt[3];
					pt[0] = clustered_points[cid][0];
					pt[1] = clustered_points[cid][1];
					pt[2] = clustered_points[cid][2];
					world2pixel(pt);


					rectangle(display, Rect(pt[0] - 2, pt[1] - 2, 4, 4), CV_RGB(r, g, b), 2);

				}
				else if (jid == R_KNEE)
				{

					cid = save_cluster_id[13];
					float dist = sqrt(pow(joint_test[f][jid].x_world - clustered_points[cid][0], 2) + pow(joint_test[f][jid].y_world - clustered_points[cid][1], 2) + pow(joint_test[f][jid].depth - clustered_points[cid][2], 2));
					if (dist <= THR)
						average_result_count[10]++;
					r = 255;
					g = 51;
					b = 102;	//pink

					float pt[3];
					pt[0] = clustered_points[cid][0];
					pt[1] = clustered_points[cid][1];
					pt[2] = clustered_points[cid][2];
					world2pixel(pt);


					rectangle(display, Rect(pt[0] - 2, pt[1] - 2, 4, 4), CV_RGB(r, g, b), 2);

				}
				else if (jid == R_ANKLE)
				{

					cid = save_cluster_id[14];
					float dist = sqrt(pow(joint_test[f][jid].x_world - clustered_points[cid][0], 2) + pow(joint_test[f][jid].y_world - clustered_points[cid][1], 2) + pow(joint_test[f][jid].depth - clustered_points[cid][2], 2));
					if (dist <= THR)
						average_result_count[11]++;
					r = 0;
					g = 51;
					b = 0;	//dark green

					float pt[3];
					pt[0] = clustered_points[cid][0];
					pt[1] = clustered_points[cid][1];
					pt[2] = clustered_points[cid][2];
					world2pixel(pt);


					rectangle(display, Rect(pt[0] - 2, pt[1] - 2, 4, 4), CV_RGB(r, g, b), 2);

				}
			}

			//upper bound for each joint
			int uid = 0;
			for (int jid = 0; jid < JOINT_NUMBER; jid++)
			{
				if (jid != TRSO && jid != L_HIPS && jid != R_HIPS && jid != BLLY)
				{
					for (int cid = 0; cid < clustered_points.size(); cid++)
					{
						float dist = sqrt(pow(joint_test[f][jid].x_world - clustered_points[cid][0], 2) + pow(joint_test[f][jid].y_world - clustered_points[cid][1], 2) + pow(joint_test[f][jid].depth - clustered_points[cid][2], 2));
						if (dist <= 0.1)
						{
							upper_bound_result_count[uid]++;
							break;
						}
					}
					uid++;
				}
			}


			////draw joint
			//for (int joint_index = 0; joint_index < JOINT_NUMBER; joint_index++)
			//	rectangle(display, Rect(joint_test[f][joint_index].x_pixel - 2, joint_test[f][joint_index].y_pixel - 2, 4, 4), CV_RGB(0, 255, 0), 2);

			ss << SAVE_DIR << f << ".jpg";
			imwrite(ss.str(), display);
			ss.str("");
			ss.clear();
			tot_end = clock();
			time_result << "TOTAL: " << (double)(tot_end - tot_begin) / CLOCKS_PER_SEC << endl;
		}
		tot_frame_end = clock();

		//fstream close
		for (int i = 0; i < RESULT_JOINT_NUMBER; i++)
		{
			ss << (float)average_result_count[i] / (float)TEST_FRAME_NUM;
			average_result << ss.str() << endl;
			ss.str("");
			ss.clear();
		}
		average_result.close();
		for (int i = 0; i < RESULT_JOINT_NUMBER; i++)
		{
			ss << (float)upper_bound_result_count[i] / (float)TEST_FRAME_NUM;
			upper_bound_result << ss.str() << endl;
			ss.str("");
			ss.clear();
		}
		upper_bound_result.close();

		time_result << "FPS: " << (double)TEST_FRAME_NUM / ((double)(tot_frame_end - tot_frame_begin) / CLOCKS_PER_SEC) << endl;
		time_result.close();
	}
	
}

void save_tree_recursion(node* tree, ofstream& of)
{
	stringstream ss;

	if (tree->left_child == NULL && tree->right_child == NULL)
	{
		ss << 0 << " " << tree->depth << " " << tree->delta[0] << " " << tree->delta[1] << " " << tree->delta[2] << " " << tree->delta[3] << " " << tree->delta[4] << " " << tree->search_radius_xy << " " << tree->search_radius_z << " " << tree->clustered_offset.size() << endl;
		for (int i = 0; i < tree->clustered_offset.size(); i++)
		{
			for (int nid = 0; nid < NEAREST_NUM; nid++)
			{
				for (int cid = 0; cid < COORDINATE_DIM; cid++)
					ss << tree->clustered_offset[i].offset[nid][cid] << " ";
			}
			ss << tree->clustered_offset[i].priority << endl;
		}

		of << ss.str();
		return;
	}
	else
	{
		ss << 1 << " " << tree->depth << " " << tree->delta[0] << " " << tree->delta[1] << " " << tree->delta[2] << " " << tree->delta[3] << " " << tree->delta[4] << " " << tree->search_radius_xy << " " << tree->search_radius_z << " " << tree->clustered_offset.size() << endl;
		of << ss.str();
		save_tree_recursion(tree->left_child, of);
		save_tree_recursion(tree->right_child, of);
	}
}
void save_tree(vector<node*> tree_head)
{
	ofstream of;
	stringstream ss;
	vector<string> file_list;


	ss << SAVE_DIR << "tree_save";
#if WINDOW
	CreateDirectory(ss.str().c_str(), NULL);
#else
	boost::filesystem::path dir(ss.str().c_str());
	boost::filesystem::create_directory(dir);
#endif
	ss.str("");
	ss.clear();


	ss << SAVE_DIR << "tree_save/";
	file_list = get_all_files_names_within_folder(ss.str());
	ss.str("");
	ss.clear();

	for (int i = 0; i < file_list.size(); i++)
	{
		ss << SAVE_DIR << "tree_save/" << file_list[i];
		remove(ss.str().c_str());
		ss.str("");
		ss.clear();
	}

	for (int i = 0; i < tree_head.size(); i++)
	{
		ss << SAVE_DIR << "tree_save/" << i << ".txt";
		of.open(ss.str());
		ss.str("");
		ss.clear();
		save_tree_recursion(tree_head.at(i), of);
		of.close();
	}

}

void load_tree_recursion(node* tree, ifstream& inf)
{
	int stop_criteria;
	int cluster_size;
	stringstream ss;
	string line;
	getline(inf, line);
	ss << line;
	ss >> stop_criteria >> tree->depth >> tree->delta[0] >> tree->delta[1] >> tree->delta[2] >> tree->delta[3] >> tree->delta[4] >> tree->search_radius_xy >> tree->search_radius_z >> cluster_size;
	if (stop_criteria == 0)
	{
		ss.str("");
		ss.clear();

		for (int i = 0; i < cluster_size; i++)
		{
			cluster tmp;
			tmp.offset = new float*[NEAREST_NUM];
			for (int nid = 0; nid < NEAREST_NUM; nid++)
				tmp.offset[nid] = new float[COORDINATE_DIM];

			getline(inf, line);
			ss << line;
			for (int nid = 0; nid < NEAREST_NUM; nid++)
			{
				for (int cid = 0; cid < COORDINATE_DIM; cid++)
					ss >> tmp.offset[nid][cid];
			}
			ss >> tmp.priority;

			tree->clustered_offset.push_back(tmp);
			ss.str("");
			ss.clear();
		}
	}
	else
	{
		tree->left_child = new node();
		tree->right_child = new node();
		load_tree_recursion(tree->left_child, inf);
		load_tree_recursion(tree->right_child, inf);
	}

}
void load_tree(vector<node*>* tree_head)
{
	ifstream inf;
	stringstream ss;
	node* tree_ptr;

	for (int i = 0;; i++)
	{
		ss << SAVE_DIR << "tree_save/" << i << ".txt";

		inf.open(ss.str());
		ss.str("");
		ss.clear();

		if (!inf)
			break;

		tree_ptr = new node();

		load_tree_recursion(tree_ptr, inf);
		tree_head->push_back(tree_ptr);
		inf.close();
	}

}
