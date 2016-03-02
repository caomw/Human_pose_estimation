#include "Tree_architecture.h"
#include "ETC.h"


int main()
{
	stringstream ss;
	clock_t begin, end, training_time;
	string is_save;
	srand(time(NULL));

	cout << "Human pose estimation CVPR2016 based on EVAL DB" << endl;
	cout << "Developed by Gyeongsik Moon" << endl << endl;

	cout << "SAVE or LOAD: ";
	cin >> is_save;
	cout << endl;

	vector<node*> tree_head;
	point *point_train;
	joint **joint_train;
	point *point_matching;
	joint **joint_matching;
	point *point_test;
	joint **joint_test;

	

	if (!is_save.compare("SAVE"))
	{
		point_train = new point[TRAIN_FRAME_NUM * Q_HEIGHT * Q_WIDTH];
		joint_train = new joint *[TRAIN_FRAME_NUM];
		for (int f = 0; f < TRAIN_FRAME_NUM; f++)
			joint_train[f] = new joint[JOINT_NUMBER];


		//Load training data
		cout << "Loading training data..." << endl;
		load_data(point_train, joint_train, TRAINING);
		cout << endl;
		
		//Train a tree
		cout << "Training start!" << endl;
		begin = clock();
		tree_train(point_train, joint_train, tree_head);
		end = clock();
		training_time = end - begin;

		cout << endl << "Training complete!" << endl;
		cout << (float)(end - begin) / CLOCKS_PER_SEC << "seconds for training" << endl << endl;

		cout << "Saving tree..." << endl;
		save_tree(tree_head);

		for (long long int pid = 0; pid < TRAIN_FRAME_NUM * Q_HEIGHT * Q_WIDTH; pid++)
		{
			if (point_train[pid].depth != BACKGROUND_DEPTH)
			{
				for (int nid = 0; nid < NEAREST_NUM; nid++)
					delete[] point_train[pid].offset[nid];

				delete[] point_train[pid].offset;
			}
			
		}
		delete[] point_train;
		for (int f = 0; f < TRAIN_FRAME_NUM; f++)
			delete[] joint_train[f];
		delete[] joint_train;
	}
	else if (!is_save.compare("LOAD"))
	{
		cout << "Loading tree..." << endl;
		training_time = 0;
		load_tree(&tree_head);
	}
	else
	{
		cout << "Not a proper input..." << endl;
		return 0;
	}
	
	point_matching = new point[POINT_MATCHING_FRAME_NUM * Q_HEIGHT * Q_WIDTH];
	joint_matching = new joint *[POINT_MATCHING_FRAME_NUM];
	for (int f = 0; f < POINT_MATCHING_FRAME_NUM; f++)
		joint_matching[f] = new joint[JOINT_NUMBER];

	cout << "Loading DB for point matching..." << endl;
	load_data(point_matching, joint_matching, POINT_MATCHING);
	
	float** center_matching;
	center_matching = new float*[POINT_MATCHING_FRAME_NUM];
	for (int matching_f = 0; matching_f < POINT_MATCHING_FRAME_NUM; matching_f++)
	{
		center_matching[matching_f] = new float[COORDINATE_DIM];
		center_matching[matching_f][0] = 0;
		center_matching[matching_f][1] = 0;
		center_matching[matching_f][2] = 0;
		float pixel_num = 0;

		for (int matching_i = 0; matching_i < Q_WIDTH * Q_HEIGHT; matching_i++)
		{
			if (point_matching[matching_f * Q_HEIGHT * Q_WIDTH + matching_i].depth != BACKGROUND_DEPTH)
			{
				center_matching[matching_f][0] += point_matching[matching_f * Q_HEIGHT * Q_WIDTH + matching_i].x_world;
				center_matching[matching_f][1] += point_matching[matching_f * Q_HEIGHT * Q_WIDTH + matching_i].y_world;
				center_matching[matching_f][2] += point_matching[matching_f * Q_HEIGHT * Q_WIDTH + matching_i].depth;
				pixel_num++;
			}
		}
		center_matching[matching_f][0] /= pixel_num;
		center_matching[matching_f][1] /= pixel_num;
		center_matching[matching_f][2] /= pixel_num;

	}
	delete[] point_matching;


	point_test = new point[TEST_FRAME_NUM * Q_HEIGHT * Q_WIDTH];
	joint_test = new joint *[TEST_FRAME_NUM];
	for (int f = 0; f < TEST_FRAME_NUM; f++)
		joint_test[f] = new joint[JOINT_NUMBER];

	//load test dataset
	cout << "Loading test dataset..." << endl;
	load_data(point_test, joint_test, TEST);

	//Test
	cout << "Testing start!" << endl;
	begin = clock();
	//tree traversal
	tree_traversal(point_test, joint_test, center_matching, joint_matching, tree_head);
	end = clock();
	cout << "Testing complete!" << endl << endl;

	cout << (float)training_time / CLOCKS_PER_SEC << "seoncds for training" << endl;
	cout << (float)(end - begin) / CLOCKS_PER_SEC << "seoncds for testing" << endl;


	for (int matching_f = 0; matching_f < POINT_MATCHING_FRAME_NUM; matching_f++)
		delete[] center_matching[matching_f];
	delete[] center_matching;
	
	for (int f = 0; f < POINT_MATCHING_FRAME_NUM; f++)
		delete[] joint_matching[f];
	delete[] joint_matching;
	

	for (int f = 0; f < TEST_FRAME_NUM; f++)
		delete[] joint_test[f];
	delete[] joint_test;
	delete[] point_test;

	return 0;
}
