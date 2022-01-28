/*
 * vi:ts=4:tw=78:shiftwidth=4:expandtab
 * vim600:fdm=marker
 */

/**
     @file one_class_svm.cpp
    @brief one class SVM
    @author Changki Lee (leeck@kangwon.ac.kr)
    @date 2012/5/9
*/
#ifdef WIN32
#pragma warning(disable: 4786)
#pragma warning(disable: 4996)
#pragma warning(disable: 4267)
#pragma warning(disable: 4244)
#pragma warning(disable: 4018)
#endif

#include <cassert>
#include <stdexcept> //for std::runtime_error
#include <memory>    //for std::bad_alloc
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <string.h>
#include <stdlib.h>

#include "one_class_svm.hpp"
#include "timer.hpp"

using namespace std;

One_Class_SVM::One_Class_SVM() {
    // pointer
    theta = NULL;
    pred_map = new map<string, int>;
    pred_vec = new vector<string>;

    n_theta = 0;
    n_event = 0;
    n_test_event = 0;

    // parameter
    verbose = 0;
    binary = 0;
    skip_eval = 0;
    train_num = 0;

    buf = 10;
    iter = 100;
    period = 0;
    tol = 1e-4;

    // svm
    kernel_type = LINEAR;
    gamma = 1;
    coef = 1;
    degree = 3;
    cache_size = 30;
    rm_inactive = 50;      // inactive const 제거 조건 (iteration)
    cost = 1;
    eps = 0.001;

    // linear이면 1e-25, non-linear이면 1e-30
    precision = (kernel_type == LINEAR ? 1e-25 : 1e-30);
    rho = 1;

    // save sv
    save_sv = 0;
    // GRAM size
    gram_size = 100;
}

One_Class_SVM::~One_Class_SVM() {
    clear();
}

void One_Class_SVM::clear() {
    n_theta = n_event = n_test_event = 0;

    train_data.clear();
    test_data.clear();

    // svm
    alpha.clear();
    alpha_history.clear();
    work_set.clear();
    loss.clear();
    x_norm_vec.clear();
    sent_ids.clear();
    work_set_ids.clear();
    opti.clear();
    slacks.clear();
    slacks_id.clear();
    cache.clear();
    cost_vec.clear();
    // gram matrix
    for (size_t i=0; i < gram.size(); i++) {
        gram[i].clear();
    }
    gram.clear();
    gram_size = 100;
}


void One_Class_SVM::load(const string model) {
    ifstream f(model.c_str());
    if (!f) {
        cerr << "Fail to open file: " << model << endl;
        exit(1);
    }

    cerr << "loading " << model << " ... ";
    timer t;

    // init
    if (pred_map != NULL) {
        delete pred_map;
    }
    pred_map = new map<string, int>;

    if (pred_vec != NULL) {
        delete pred_vec;
    }
    pred_vec = new vector<string>;

    n_theta = 0;

    int count, fid;
    int i;
    string line;

    // check model format
    getline(f, line);
    if (strncmp(line.c_str(), "#txt,one_class_svm", 18)) {
        cerr << "Model format error: not txt model!" << endl;
        char temp[100];
        strncpy(temp, line.c_str(), 18);
        temp[18] = 0;
        cerr << temp << endl;
        exit(1);
    }

    // read context predicates
    getline(f, line);
    count = atoi(line.c_str());
    for (i = 0; i < count; ++i) {
        getline(f, line);
        (*pred_map)[line] = i;
        (*pred_vec).push_back(line);
    }
    cerr << "(pred_map:" << t.elapsed() << ") ";

    // load theta
    getline(f, line);
    n_theta = atoi(line.c_str());

    if (theta != NULL) {
        delete[] theta;
    }
    theta = new float[n_theta];

    i = 0;
    while (getline(f, line)) {
        assert(!line.empty());
        theta[i] = atof(line.c_str());
        i++;
    }
    assert(i == n_theta);

    // 소요 시간 출력
    cerr << "(" << t.elapsed() << ") done."  << endl;
}

void One_Class_SVM::load_bin(const string model) {
    FILE *f;
    f = fopen(model.c_str(), "rb");

    if (!f) {
        cerr << "Fail to open file: " << model << endl;
        exit(1);
    }

    cerr << "loading " << model << " ... ";
    timer t;

    // init
    if (pred_map != NULL) {
        delete pred_map;
    }
    pred_map = new map<string, int>;

    if (pred_vec != NULL) {
        delete pred_vec;
    }
    pred_vec = new vector<string>;

    n_theta = 0;

    int count, fid, len, i, j;
    // 너무 많이 잡으면 win32에서 에러가 남
    char buffer[1024*16];

    // skip header
    //int header_len = sizeof("bin,maxent") - 1;
    //fseek(f, header_len, 0);
    // check model format
    fread((void*)&buffer, sizeof("#bin,one_class_svm"), 1u, f);
    if (strncmp(buffer, "#bin,one_class_svm", sizeof("#bin,one_class_svm"))) {
        cerr << "Model format error: not binary model!" << endl;
        buffer[sizeof("#bin,one_class_svm")] = 0;
        cerr << buffer << endl;
        exit(1);
    }

    // read context predicates
    fread((void*)&count, sizeof(count), 1u, f);
    for (i = 0; i < count; ++i) {
        fread((void*)&len, sizeof(len), 1u, f);
        fread((void*)&buffer, len, 1u, f);
        string line(buffer, len);
        (*pred_map)[line] = i;
        (*pred_vec).push_back(line);
    }
    cerr << "(pred_map:" << t.elapsed() << ") ";

    // load theta
    fread((void*)&n_theta, sizeof(n_theta), 1u, f);

    if (theta != NULL) {
        delete[] theta;
    }
    theta = new float[n_theta];

    float theta_i;
    for (i = 0; i < n_theta; ++i) {
        fread((void*)&theta_i, sizeof(float), 1u, f);
        theta[i] = theta_i;
    }

    fclose(f);

    // 소요 시간 출력
    cerr << "(" << t.elapsed() << ") done."  << endl;
}


void One_Class_SVM::load_support_vector(const string model) {
    ifstream f(model.c_str());

    if (!f) {
        cerr << "Fail to open file: " << model << endl;
        exit(1);
    }

    cerr << "loading " << model << " ... ";
    timer t;

    int sv_num, i, j;
    string line;

    // skip header comments
    getline(f, line);
    while (line.empty() || line[0] == '#')
        getline(f, line);

    // read kernel param
    kernel_type = atoi(line.c_str());
    getline(f, line);
    gamma = atof(line.c_str());
    getline(f, line);
    coef = atof(line.c_str());
    getline(f, line);
    degree = atoi(line.c_str());

    // load support vector
    getline(f, line);
    sv_num = atoi(line.c_str());
    for (i=0; i < sv_num; i++) {
        vect_t vect;
        getline(f, line);
        vector<string> tokens;
        tokenize(line, tokens, " ");
        double cur_alpha = atof(tokens[0].c_str());
        for (j=1; j < tokens.size(); j++) {
            string fi = tokens[j];
            if (fi.find(":") != string::npos) {
                vector<string> str_vec;
                split(fi, str_vec, ":");
                feature_t feature;
                feature.pid = atoi(str_vec[0].c_str());
                feature.fval = (float)atof(str_vec[1].c_str());
                vect.feature.push_back(feature);
            } else {
                cerr << "Error: load_support_vector(): " << fi << endl;
                exit(1);
            }
        }
        // alpha & work_set
        alpha.push_back(cur_alpha);
        work_set.push_back(vect);
        // 나머지 필요 정보 초기화
        alpha_history.push_back(0);
        x_norm_vec.push_back(kernel(vect, vect));
    }

    // load sent_ids
    getline(f, line);
    if (sv_num != atoi(line.c_str())) {
        cerr << "Error: load_support_vector!" << endl;
        exit(1);
    }
    for (i=0; i < sv_num; i++) {
        getline(f, line);
        sent_ids.push_back(atoi(line.c_str()));
    }

    // load train_data_size
    getline(f, line);
    int train_data_size = atoi(line.c_str());

    // work_set_ids 작성
    for (i=0; i < train_data_size; i++) {
        vector<int> temp;
        work_set_ids.push_back(temp);
    }
    for (i=0; i < sent_ids.size(); i++) {
        int id = sent_ids[i];
        work_set_ids[id].push_back(i);
    }

    // 소요 시간 출력
    cerr << "(" << t.elapsed() << ") done."  << endl;
}


void One_Class_SVM::save(const string model) {
    int i, j;

    ofstream f(model.c_str());
    f.precision(8);

    if (!f) {
        cerr << "Unable to open model file to write: " << model << endl;
        exit(1);
    }

    // todo: write a header section here
    f << "#txt,one_class_svm" << endl;

    f << (*pred_vec).size() << endl;
    for (i = 0; i < (*pred_vec).size(); ++i)
        f << (*pred_vec)[i] << endl;

    // write theta
    f << n_theta << endl;
    for (i = 0; i < n_theta; ++i)
        f << theta[i] << endl;
}

void One_Class_SVM::save_bin(const string model) {
    FILE *f;
    f = fopen(model.c_str(), "wb");

    if (!f) {
        cerr << "Unable to open model file to write: " << model << endl;
        exit(1);
    }

    int i, j, uint;

    // todo: write a header section here
    int header_len = sizeof("#bin,one_class_svm");
    fwrite((void*)"#bin,one_class_svm", header_len, 1u, f);

    uint = (*pred_vec).size();
    fwrite((void*)&uint, sizeof(uint), 1u, f);
    for (i = 0; i < (*pred_vec).size(); ++i) {
        uint = (*pred_vec)[i].size();
        fwrite((void*)&uint, sizeof(uint), 1u, f);
        fwrite((void*)(*pred_vec)[i].c_str(), (*pred_vec)[i].size(), 1u, f);
    }

    // write theta
    uint = n_theta;
    fwrite((void*)&uint, sizeof(uint), 1u, f);
    float theta_i;
    for (i = 0; i < n_theta; ++i) {
        theta_i = theta[i];
        fwrite((void*)&theta_i, sizeof(float), 1u, f);
    }

    fclose(f);
}

// save support vector for fsmo
void One_Class_SVM::save_support_vector(const string model) {
    int i, j, k;

    ofstream f(model.c_str());
    f.precision(8);

    if (!f) {
        cerr << "Unable to open model file to write: " << model << endl;
        exit(1);
    }

    // todo: write a header section here
    f << "#txt,one_class_svm" << endl;

    // kernel parameter
    f << kernel_type << endl;
    f << gamma << endl;
    f << coef << endl;
    f << degree << endl;

    // write support vector 
    int sv_num = 0;
    for (i=0; i < work_set.size(); i++) {
        if (alpha[i] > precision) {
            for (j=0; j < work_set[i].feature.size(); j++) {
                sv_num++;
            }
        }
    }
    f << sv_num << endl;
    for (i=0; i < work_set.size(); i++) {
        if (alpha[i] > precision) {
            for (j=0; j < work_set[i].feature.size(); j++) {
                f << " " << work_set[i].feature[j].pid << ":" << work_set[i].feature[j].fval;
            }
            f << endl;
        }
    }

    // write sent_ids
    f << sv_num << endl;
    for (i=0; i < work_set.size(); i++) {
        if (alpha[i] > precision) {
            f << sent_ids[i];
            f << endl;
        }
    }

    // write train_data.size()
    f << train_data.size() << endl;
}

// show feature weight
void One_Class_SVM::show_feature() {
    for (size_t pid = 0; pid < (*pred_vec).size(); pid++) {
        cout << pid << "\t" << (*pred_vec)[pid] << "\t";
        cout << theta[pid] << endl;
    }
}


// remove zero feature
void One_Class_SVM::remove_zero_feature(double threshold) {
    // hash crf인 경우 skip
    if ((*pred_vec).empty()) return;

    map<string, int> *new_pred_map = new map<string, int>;
    vector<string> *new_pred_vec = new vector<string>;
    float *new_theta = new float[n_theta];

    int new_pid = 0, remove_count = 0;

    for (size_t pid = 0; pid < (*pred_vec).size(); pid++) {
        if (theta[pid] < threshold) {
            remove_count++;
            continue;
        }
        string pred = (*pred_vec)[pid];
        (*new_pred_map)[pred] = (*new_pred_vec).size();
        (*new_pred_vec).push_back(pred);
        new_theta[new_pid++] = theta[pid];
    }

    cerr << "remove zero feature (threshold=" << threshold << "):" << endl;
    cerr << "\t" << remove_count << " pred (" << 100.0*remove_count/(*pred_vec).size() << "%): " << (*pred_vec).size() << " --> " << (*new_pred_vec).size() << endl;

    // theta 정확한 사이즈로
    n_theta = new_pid;
    delete[] theta;
    theta = new float[n_theta];
    for (int i = 0; i < n_theta; i++) {
        theta[i] = new_theta[i];
    }
    delete[] new_theta;

    delete pred_map;
    delete pred_vec;

    pred_map = new_pred_map;
    pred_vec = new_pred_vec;
}


// load event
void One_Class_SVM::load_event(const string file) {
    string line;
    int i, j, count = 0;
    int old_n_theta = n_theta;
    int old_pred_vec_size = (*pred_vec).size();

    ifstream f(file.c_str());
    if (!f) {
        cerr << "Can not open data file to read: " << file << endl;
        exit(1);
    }

    while (getline(f, line)) {
        // remove newline for windows format file
        string find_str = "\r";
        string::size_type find_pos = line.find(find_str);
        if (string::npos != find_pos) {
            line.replace(find_pos, find_str.size(), "");
        }
        if (line.empty()) {
            continue;
        } else if (line[0] == '#') {
            // comment 처리
        } else {
            /// Tokenizer (stardust)
            vector<string> tokens;
            tokenize(line, tokens, " \t");
            vector<string>::iterator it = tokens.begin();

            data_t data;
            data.outcome = 1;

            it++;
            for (; it != tokens.end(); it++) {
                string fi(it->c_str());
                // test
                //cerr << " " << *it;
                // pred
                int pid;
                bool new_pid = false;
                // fval
                float fval = 1.0;
                if (fi.find(":") != string::npos) {
                    vector<string> str_vec;
                    split(fi, str_vec, ":");
                    fi = str_vec[0];
                    fval = atof(str_vec[1].c_str());
                }

                if ((*pred_map).find(fi) == (*pred_map).end()) {
                    new_pid = true;
                    if (1 /* !incremental || !support_feature */) {
                        pid = (*pred_vec).size();
                        (*pred_map)[fi] = pid;
                        (*pred_vec).push_back(fi);
                    } else if (1) {
                        cerr << "skip feature:" << fi << " ";
                    }
                } else {
                    new_pid = false;
                    pid = (*pred_map)[fi];
                }

                feature_t feature;
                feature.pid = pid;
                feature.fval = fval;
                data.fvec.feature.push_back(feature);
            }
            data.fvec.factor = 1;
            data.fvec.twonorm_sq = dot_product(data.fvec, data.fvec);

            train_data.push_back(data);
            if (train_num > 0 && train_num <= train_data.size()) {
                ++count;
                break;
            }

            ++count;
            if (count % 10000 == 0) {
                cerr << ".";
                if (count % 100000 == 0)
                    cerr << " ";
                if (count % 500000 == 0)
                    cerr << "\t" << count << endl;
            }
        }
    }

    n_event += count;

    n_theta = (*pred_vec).size();
    if (theta != NULL) delete[] theta;

    // 메모리 할당
    cerr << endl << "theta allocated: " << n_theta << " ... ";
    theta = new float[n_theta];
    cerr << "Done." << endl;

    for (i=0; i < n_theta; i++) {
        theta[i] = 0.0;
    }
}


void One_Class_SVM::load_test_event(const string file) {
    string line;
    int count = 0;

    ifstream f(file.c_str());
    if (!f) {
        cerr << "Can not open data file to read: " << file << endl;
        exit(1);
    }

    test_data.clear();

    while (getline(f, line)) {
        // remove newline for windows format file
        string find_str = "\r";
        string::size_type find_pos = line.find(find_str);
        if (string::npos != find_pos) {
            line.replace(find_pos, find_str.size(), "");
        }
        if (line.empty()) {
            continue;
        } else if (line[0] == '#') {
            // comment 처리 안함
        } else {
            /// Tokenizer (stardust)
            vector<string> tokens;
            tokenize(line, tokens, " \t");
            vector<string>::iterator it = tokens.begin();

            int oid;
            if (*it == "1" || *it == "+1") oid = 1;
            else if (*it == "-1") oid = -1;
            else {
                if (verbose) cerr << "Warning: output: " << *it << endl;
                oid = -1;
            }

            data_t data;
            data.outcome = oid;

            ++it;
            for (; it != tokens.end();) {
                string fi(it->c_str()); ++it;

                feature_t feature;
                feature.fval = 1;
                if (fi.find(":") != string::npos) {
                    vector<string> str_vec;
                    split(fi, str_vec, ":");
                    fi = str_vec[0];
                    feature.fval = atof(str_vec[1].c_str());
                }

                if ((*pred_map).find(fi) != (*pred_map).end()) {
                    feature.pid = (*pred_map)[fi];
                    data.fvec.feature.push_back(feature);
                } else if (verbose) {
                    //cerr << "Warning: " << fi << endl;
                }
            }
            data.fvec.factor = 1;
            data.fvec.twonorm_sq = dot_product(data.fvec, data.fvec);

            test_data.push_back(data);

            ++count;
            if (count % 10000 == 0) {
                cerr << ".";
                if (count % 100000 == 0)
                    cerr << " ";
                if (count % 500000 == 0)
                    cerr << "\t" << count << endl;
            }
        }
    }

    n_test_event = count;
}


// random_shuffle train_data
void One_Class_SVM::random_shuffle_train_data() {
    random_shuffle(train_data.begin(), train_data.end());
}


int One_Class_SVM::predict(ostream& f) {
    f.precision(8);
    vector<data_t>::iterator it = test_data.begin();
    int correct = 0;
    int total = 0;
    int i = 0;

    // 파일에 쓰기
    f << "# test answer score feature" << endl; 
    timer t;

    for (; it != test_data.end(); ++it) {
        data_t& data = *it;
        int y = 1;
        double score = eval(data.fvec);
        if (score < rho) y = -1;

        total++;
        if (data.outcome == y) {
            correct++;
            f << "O " << data.outcome << " " << score - rho;
        } else {
            f << "X " << data.outcome << " " << score - rho;
        }
        // feature 출력
        for (int i=0; i < data.fvec.feature.size(); i++) {
            int pid = data.fvec.feature[i].pid;
            double fval = data.fvec.feature[i].fval;
            f << " " << (*pred_vec)[pid] << ":" << fval;
        }
        f << endl;
    }
    cout << "Accuracy: " << 100.0 * correct / total << "%\t("
        << correct << "/" << total << ")" << endl;

    // 소요 시간 출력
    cout << t.elapsed() << " sec, " << n_test_event / t.elapsed() << " tokens per sec (" << n_test_event << " / " << t.elapsed() << ")" << endl;

    return correct;
}


// eval (rho는 계산 안함) : LINEAR: w * x or sum {alpha * sv * x}
double One_Class_SVM::eval(vect_t& vect) {
    double result = 0;
    if (kernel_type == LINEAR) {
        for (int i=0; i < vect.feature.size(); i++) {
            result += theta[vect.feature[i].pid] * vect.feature[i].fval;
        }
    } else {
        for (int i=0; i < alpha.size(); i++) {
            if (alpha[i] > precision) {
                result += alpha[i] * kernel(work_set[i], vect);
            }
        }
    }
    // w * x
    return vect.factor * result;
}


// make_data
data_t One_Class_SVM::make_data(const vector<string>& context) {
    data_t data;
    data.outcome = 0;
    map<string, int> &pred_map_ref = *pred_map;

    for (int j=0; j < context.size(); j++) {
        string fi = context[j];
        feature_t feature;
        feature.fval = 1;
        if (fi.find(":") != string::npos) {
            vector<string> str_vec;
            split(fi, str_vec, ":");
            fi = str_vec[0];
            feature.fval = atof(str_vec[1].c_str());
        }
        if (pred_map_ref.find(fi) != pred_map_ref.end()) {
            feature.pid = pred_map_ref[fi];
            data.fvec.feature.push_back(feature);
        }
    }
    data.fvec.factor = 1;
    data.fvec.twonorm_sq = dot_product(data.fvec, data.fvec);

    return data;
}


// dot_product
double One_Class_SVM::dot_product(vect_t& vect1, vect_t& vect2) {
    double result = 0.0;
    vector<feature_t>::iterator it1 = vect1.feature.begin();
    vector<feature_t>::iterator it1_end = vect1.feature.end();
    vector<feature_t>::iterator it2 = vect2.feature.begin();
    vector<feature_t>::iterator it2_end = vect2.feature.end();

    while (it1 != it1_end && it2 != it2_end) {
        if (it1->pid == it2->pid) {
            result += it1->fval * it2->fval;
            it1++;
            it2++;
        } else if (it1->pid > it2->pid) {
            it2++;
        } else {
            it1++;
        }
    }

    return vect1.factor * vect2.factor * result;
}


// kernel function : for cache
double One_Class_SVM::kernel(int v1, int v2) {
    double prod;
    int key;

    // a * b = b * a 라는 가정
    if (v1 > v2) {
        key = v1 * train_data.size() * 11 + v2;
    } else {
        key = v2 * train_data.size() * 11 + v1;
    }

    int real_cache_size = cache_size * 1024 * 1024 / (3 * sizeof(int) + sizeof(float));

    if (cache.find(key) != cache.end()) {
        // test
        //cerr << ".";
        prod = cache[key];
        // support vector만 cache
        if (cache.size() > (3 * real_cache_size / 5) && (alpha[v1] == 0 || alpha[v2] == 0)) {
            cache.erase(key);
            //cerr << "cache erase:" << cache.size() << endl;
        }
    } else {
        // test
        //cerr << "+";
        prod = kernel(work_set[v1], work_set[v2]);
        // support vector만 cache
        if (alpha[v1] > 0 && alpha[v2] > 0) {
            if (cache.size() < real_cache_size) {
                cache[key] = prod;
            } else {
                //cerr << endl << "@@@ cache insufficient: " << cache.size() << endl;
            }
        }
    }

    return prod;
}

// kernel function
double One_Class_SVM::kernel(vect_t& vect1, vect_t& vect2) {
    double prod = dot_product(vect1, vect2);

    switch (kernel_type) {
        case LINEAR:
            return prod;
        case POLY:
            return pow(gamma * prod + coef, degree);
        case RBF:
            return exp(-gamma * (vect1.twonorm_sq + vect2.twonorm_sq - 2*prod));
        case SIGMOID:
            return tanh(gamma * prod + coef);
        case USER:
            return 1.0 / (gamma * (vect1.twonorm_sq + vect2.twonorm_sq - 2*prod) + coef);
        default:
            cerr << "kernel Error! : " << kernel_type << endl;
            exit(1);
    }
}



/** Tokenize string to words.
 Tokenization of string and assignment to word vector.
 Delimiters are set of char.
 @param str string
 @param tokens token vector
 @param delimiters delimiters to divide string
 @return none
 */
void One_Class_SVM::tokenize(const string& str, vector<string>& tokens, const string& delimiters) {
    tokens.clear();
    string::size_type lastPos = str.find_first_not_of(delimiters, 0);
    string::size_type pos  = str.find_first_of(delimiters, lastPos);
    while (string::npos != pos || string::npos != lastPos) {
        tokens.push_back(str.substr(lastPos, pos - lastPos));
        lastPos = str.find_first_not_of(delimiters, pos);
        pos = str.find_first_of(delimiters, lastPos);
    }
} 

/** Split string to words.
 Tokenization of string and assignment to word vector.
 Delimiter is string.
 @param str string
 @param tokens token vector
 @param delimiter delimiter to divide string
 @return none
 */
void One_Class_SVM::split(const string& str, vector<string>& tokens, const string& delimiter) {
    tokens.clear();
    string::size_type pos = str.find(delimiter, 0);
    string::size_type lastPos = 0;
    while (0 <= pos && str.size() > pos) {
        tokens.push_back(str.substr(lastPos, pos - lastPos));
        lastPos = pos + delimiter.size();
        pos = str.find(delimiter, lastPos);
    }
    tokens.push_back(str.substr(lastPos, str.size() - lastPos));
} 


