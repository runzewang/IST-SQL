# IST-SQL
Codes for AAAI2021 paper “Tracking Interaction States for Multi-Turn Text-to-SQL Semantic Parsing”

Training step:

1. download the data from https://drive.google.com/drive/folders/1LRt_sLxChAkkLOf0ZrAh2-PJTX0HYF9I?usp=sharing , put the folder under the same path of the code folder. 
2. download the pretrained bert model from https://drive.google.com/drive/folders/1De1tyughemVMem-aGYTCo076GZxZvVCS?usp=sharing, put the folder under the same path of the code folder. 
3. check the input and output paths in the preprocess.py file.
4. check the input and output paths in the run_cosql_editsql.sh file.
5. to run: sh cosql_train.sh [gpu_id] [editsql] [save_name]

Testing step:
1. download the trained model from https://drive.google.com/drive/folders/1-TdfSMSKYY8osHFOlUEb6hkUmjKEPf_r?usp=sharing, and put it under the code folder.
2. check the input and output path in the test_cosql_editsql.sh
3. to test: sh test_cosql_editsql.sh

Acknowledge:
This project is conducted based on the project https://github.com/ryanzhumich/editsql

Citation:
If you use this project, please cite our paper:
Wang, R.-Z., Ling, Z.-H., Zhou, J., & Hu, Y. (2021). Tracking Interaction States for Multi-Turn Text-to-SQL Semantic Parsing. Proceedings of the AAAI Conference on Artificial Intelligence, 35(16), 13979-13987. Retrieved from https://ojs.aaai.org/index.php/AAAI/article/view/17646
