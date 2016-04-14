
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
	int joint_fid = 0;
	int frame_id = 0;
	int iter_max = 0;
	
	stringstream ss;
	ifstream inf;

	if (mode == TEST)
	{
		frame_num = TEST_FRAME_NUM;
		iter_max = 1;
	}
	else if (mode == TRAINING)
	{
		frame_num = TRAIN_FRAME_NUM;
		iter_max = 4;
	}
	else if (mode == POINT_MATCHING)
	{
		frame_num = POINT_MATCHING_FRAME_NUM;
		iter_max = 4;
	}
	
	for (int i = 0; i < frame_num * Q_HEIGHT * Q_WIDTH; i++)
	{
		
		int frame_id = i / (Q_HEIGHT * Q_WIDTH);
		pp_point[i].frame_id = frame_id;

		float pt[3];
		pt[0] = (i - frame_id * (Q_HEIGHT * Q_WIDTH)) % Q_WIDTH;
		pt[1] = (i - frame_id * (Q_HEIGHT * Q_WIDTH)) / Q_WIDTH;
		pp_point[i].x_pixel = pt[0];
		pp_point[i].y_pixel = pt[1];
		pp_point[i].depth = BACKGROUND_DEPTH;
	}


	for (int tid = 0; tid < iter_max; tid++)
	{
		if (tid == 0)
			ss << TRAINING_DIR << TRAIN_MODEL_1 << TEST_DIFFICULTY;
		else if (tid == 1)
			ss << TRAINING_DIR << TRAIN_MODEL_2 << TEST_DIFFICULTY;
		else if (tid == 2)
			ss << TRAINING_DIR << TRAIN_MODEL_3 << TEST_DIFFICULTY;
		else if (tid == 3)
			ss << TRAINING_DIR << TRAIN_MODEL_4 << TEST_DIFFICULTY;

		if (tid == 0 && mode == TEST)
			ss << TRAINING_DIR << TEST_MODEL << TEST_DIFFICULTY;

		string cur_dir = ss.str();
		ss << "/parsed/";
		string cur_depth_dir = ss.str();
		ss.str("");
		ss.clear();

		vector<string> file_list = get_all_files_names_within_folder(cur_depth_dir);
		vector<int> fid_list;

#define RANDOM_SAMPLING_TRAINING 500
		for (int iter = 0; iter < RANDOM_SAMPLING_TRAINING; iter++)
		{

			int fid = rand() % file_list.size();
			fid_list.push_back(fid);

			ss << cur_depth_dir << file_list[fid];
			inf.open(ss.str());
			ss.str("");
			ss.clear();

			string line;
			int x_pixel, y_pixel;
			double x_world, y_world, depth;
			while (getline(inf, line))
			{
				ss << line;
				ss >> y_pixel >> x_pixel >> y_world >> x_world >> depth;

				ss.str("");
				ss.clear();

				pp_point[frame_id * Q_HEIGHT * Q_WIDTH + y_pixel * Q_WIDTH + x_pixel].x_world = x_world;
				pp_point[frame_id * Q_HEIGHT * Q_WIDTH + y_pixel * Q_WIDTH + x_pixel].y_world = y_world;
				pp_point[frame_id * Q_HEIGHT * Q_WIDTH + y_pixel * Q_WIDTH + x_pixel].depth = depth;
			}

			cout << "frame_id: " << frame_id << endl;
			frame_id++;
			inf.close();
		}
		file_list.clear();

		ss << cur_dir << "/joints.txt";
		inf.open(ss.str());
		ss.str("");
		ss.clear();
		
		string line;
		vector<string> joint_raw_data;

		while (getline(inf, line))
			joint_raw_data.push_back(line);
		
		inf.close();

		for (int iter = 0; iter < RANDOM_SAMPLING_TRAINING; iter++)
		{
			ss << joint_raw_data[fid_list[iter]];

			int effective_jid = 0;
			for (int jid = 0; jid < 20; jid++)
			{
				double x_world, y_world, depth, ori_1, ori_2, ori_3, ori_4;
				ss >> x_world >> y_world >> depth >> ori_1 >> ori_2 >> ori_3 >> ori_4;
				
				if (jid == 1 || jid == 2 || jid == 3 || jid == 4 || jid == 5 || jid == 6 || jid == 8 || jid == 9 || jid == 10 || jid == 12 || jid == 13 || jid == 14 || jid == 16 || jid == 17 || jid == 18)
				{
					pp_joint[joint_fid][effective_jid].x_world = x_world;
					pp_joint[joint_fid][effective_jid].y_world = y_world;
					pp_joint[joint_fid][effective_jid].depth = depth;

					effective_jid++;
				}
			}
			ss.str("");
			ss.clear();
			joint_fid++;
		}

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
				image->data[p * 3 + 0] = unsigned char(gray);
				image->data[p * 3 + 1] = unsigned char(gray);
				image->data[p * 3 + 2] = unsigned char(gray);
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

