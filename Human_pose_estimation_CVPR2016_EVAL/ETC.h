
#if !WINDOW
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>

vector<string> get_all_files_names_within_folder(string folder) {
	vector<string> names;
	DIR *dp;
	struct dirent *dirp;
	struct stat filestat;

	if ((dp = opendir(folder.c_str())) == NULL) {
		cout << "error in opening files" << endl;
		return vector<string>();
	}

	while ((dirp = readdir(dp)) != NULL) {
		if (stat(folder.c_str(), &filestat)) continue;
		if (S_ISDIR(filestat.st_mode)) continue;
		printf("%s\n", dirp->d_name);
		names.push_back(string(dirp->d_name));
	}
	closedir(dp);
	return names;
}



#else
vector<string> get_all_files_names_within_folder(string folder)
{
	vector<string> names;
	char search_path[1000];
	sprintf_s(search_path, 1000, "%s*.*", folder.c_str());
	WIN32_FIND_DATA fd;
	HANDLE hFind = ::FindFirstFile(search_path, &fd);
	if (hFind != INVALID_HANDLE_VALUE) {
		do {
			// read all (real) files in current folder
			// , delete '!' read other 2 default folder . and ..
			if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
				names.push_back(fd.cFileName);
			}
		} while (::FindNextFile(hFind, &fd));
		::FindClose(hFind);
	}
	return names;
}

#endif


void get_user_depth(point *pp_depth, float *depth, float *user, int frame_id)
{

	for (int j = 0; j < Q_HEIGHT; j++)
	{
		for (int i = 0; i < Q_WIDTH; i++)
		{
			if (user[j * Q_WIDTH + i] > 0)
			{
				pp_depth[Q_HEIGHT * Q_WIDTH * frame_id + Q_WIDTH * j + i].depth = depth[j * Q_WIDTH + i];
				float pt[3];
				pt[0] = pp_depth[Q_HEIGHT * Q_WIDTH * frame_id + Q_WIDTH * j + i].x_pixel;
				pt[1] = pp_depth[Q_HEIGHT * Q_WIDTH * frame_id + Q_WIDTH * j + i].y_pixel;
				pt[2] = pp_depth[Q_HEIGHT * Q_WIDTH * frame_id + Q_WIDTH * j + i].depth;
				pixel2world(pt);
				pp_depth[Q_HEIGHT * Q_WIDTH * frame_id + Q_WIDTH * j + i].x_world = pt[0];
				pp_depth[Q_HEIGHT * Q_WIDTH * frame_id + Q_WIDTH * j + i].y_world = pt[1];	
			}
			else
			{
				pp_depth[Q_HEIGHT * Q_WIDTH * frame_id + Q_WIDTH * j + i].depth = BACKGROUND_DEPTH;
				float pt[3];
				pt[0] = pp_depth[Q_HEIGHT * Q_WIDTH * frame_id + Q_WIDTH * j + i].x_pixel;
				pt[1] = pp_depth[Q_HEIGHT * Q_WIDTH * frame_id + Q_WIDTH * j + i].y_pixel;
				pt[2] = pp_depth[Q_HEIGHT * Q_WIDTH * frame_id + Q_WIDTH * j + i].depth;
				pixel2world(pt);
				pp_depth[Q_HEIGHT * Q_WIDTH * frame_id + Q_WIDTH * j + i].x_world = pt[0];
				pp_depth[Q_HEIGHT * Q_WIDTH * frame_id + Q_WIDTH * j + i].y_world = pt[1];
			}
		}
	}

}
void load_data(point* pp_point, joint** pp_joint, int mode)
{
	int file_idx = 0;
	int frame_num = 0;
	stringstream ss;

	if (mode == TEST)
		frame_num = TEST_FRAME_NUM;
	else if (mode == TRAINING)
		frame_num = TRAIN_FRAME_NUM;
	else if (mode == POINT_MATCHING)
		frame_num = POINT_MATCHING_FRAME_NUM;

	for (int i = 0; i < frame_num * Q_HEIGHT * Q_WIDTH; i++)
	{
		int frame_id = i / (Q_HEIGHT * Q_WIDTH);
		pp_point[i].frame_id = frame_id;

		float pt[3];
		pt[0] = (i - frame_id * (Q_HEIGHT * Q_WIDTH)) % Q_WIDTH;
		pt[1] = (i - frame_id * (Q_HEIGHT * Q_WIDTH)) / Q_WIDTH;
		pp_point[i].x_pixel = pt[0];
		pp_point[i].y_pixel = pt[1];
	}

	for (int frame_id = 0; frame_id < frame_num;)
	{
		FILE *ifp;
		int stats[6];

		ss << TRAINING_DIR << "eccv12_ordered_stat_" << file_idx << ".bin";
		ifp = fopen(ss.str().c_str(), "rb");
		fread(&stats, sizeof(int), 6, ifp);
		fclose(ifp);
		ss.str("");
		ss.clear();

		bool check_model = false;
		if (mode == TRAINING || mode == POINT_MATCHING)
		{
			if (stats[0] == TRAIN_MODEL_1 || stats[0] == TRAIN_MODEL_2)
				check_model = true;
		}
		else
		{
			if (stats[0] == TEST_MODEL)
				check_model = true;
		}

		if (check_model)	// stat model, action, action frame, total frame, good or bad, order idx
		{

			float *depth = new float[Q_WIDTH*Q_HEIGHT];
			float *user = new float[Q_WIDTH*Q_HEIGHT];
			float joints[JOINT_NUMBER * COORDINATE_DIM];
			float loaded_float[Q_WIDTH*Q_HEIGHT];


			ss << TRAINING_DIR << "eccv12_ordered_depth_" << file_idx << ".bin";
			ifp = fopen(ss.str().c_str(), "rb");
			fread(&loaded_float, sizeof(float), Q_WIDTH*Q_HEIGHT, ifp);
			for (int p = 0; p < Q_WIDTH*Q_HEIGHT; p++){ depth[p] = loaded_float[p]; }
			fclose(ifp);
			ss.str("");
			ss.clear();

			ss << TRAINING_DIR << "eccv12_ordered_user_" << file_idx << ".bin";
			ifp = fopen(ss.str().c_str(), "rb");
			fread(&loaded_float, sizeof(float), Q_WIDTH*Q_HEIGHT, ifp);
			for (int p = 0; p < Q_WIDTH*Q_HEIGHT; p++){ user[p] = loaded_float[p]; }
			fclose(ifp);
			ss.str("");
			ss.clear();

			get_user_depth(pp_point, depth, user, frame_id);


			ss << TRAINING_DIR << "eccv12_ordered_joint_" << file_idx << ".bin";
			ifp = fopen(ss.str().c_str(), "rb");
			fread(&joints, sizeof(float), JOINT_NUMBER * COORDINATE_DIM, ifp);
			fclose(ifp);
			ss.str("");
			ss.clear();

			for (int j = 0; j < JOINT_NUMBER; j++)
			{
				float pt[3];
				pt[0] = joints[j * COORDINATE_DIM + 0];
				pt[1] = joints[j * COORDINATE_DIM + 1];
				pt[2] = joints[j * COORDINATE_DIM + 2];
				pp_joint[frame_id][j].x_pixel = pt[0];
				pp_joint[frame_id][j].y_pixel = pt[1];

				pixel2world(pt);
				pp_joint[frame_id][j].x_world = pt[0];
				pp_joint[frame_id][j].y_world = pt[1];
				pp_joint[frame_id][j].depth = pt[2];
			}


			delete[] depth;
			delete[] user;

			frame_id++;
		}
		file_idx++;
	}
}
void draw_depth_user(Mat *image, float *depth)
{

	float pmin, pmax;
	pmin = FLT_MAX;
	pmax = -FLT_MAX;
	for (int y = 0; y < Q_HEIGHT; y++){
		for (int x = 75; x < Q_WIDTH; x++){
			int p = y*Q_WIDTH + x;
			if (depth[p] < BACKGROUND_DEPTH){
				if (pmin > depth[p]){
					pmin = depth[p];
				}
				if (pmax < depth[p]){
					pmax = depth[p];
				}
			}
		}
	}

	for (int y = 0; y < Q_HEIGHT; y++){
		for (int x = 0; x < Q_WIDTH; x++){
			int p = y*Q_WIDTH + x;
			if (depth[p] < BACKGROUND_DEPTH){
				float gray = depth[p];
				gray = (175 * (gray - pmin) / (pmax - pmin) + 50);
				image->data[p * 3 + 0] = (unsigned char)(gray);
				image->data[p * 3 + 1] = (unsigned char)(gray);
				image->data[p * 3 + 2] = (unsigned char)(gray);
			}
			else{
				image->data[p * 3 + 0] = 0xff;
				image->data[p * 3 + 1] = 0xff;
				image->data[p * 3 + 2] = 0xff;
			}
		}
	}
}

inline float rand_float(float min, float max)
{
	return (float(rand()) / float(RAND_MAX))*(max - min) + min;
}
inline void pixel2world(float pt[3])
{
	pt[0] = (pt[0] - 160.0f)*pt[2] * 0.003873f;
	pt[1] = (pt[1] - 120.0f)*pt[2] * 0.003873f;
}
inline void world2pixel(float pt[3])
{
	
	pt[0] = 160 + pt[0] * 258.2f / pt[2];
	pt[1] = 120 + (pt[1] * 258.2 / pt[2]);
}

