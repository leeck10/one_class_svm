/*
 * vi:ts=4:tw=78:shiftwidth=4:expandtab
 * vim600:fdm=marker
 */

/**
 	@file one_class_svm.hpp
	@brief one class SVM
	@author Changki Lee (leeck@kangwon.ac.kr)
	@date 2012/5/9
*/
#ifndef ONE_CLASS_SVM_H
#define ONE_CLASS_SVM_H

#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <fstream>

#include <cassert>
#include <cfloat>
#include <cmath>
#include <limits>
#include <algorithm>

// for SVM kernel type
# define LINEAR  0           /* linear kernel type */
# define POLY    1           /* polynomial kernel type */
# define RBF     2           /* rbf kernel type */
# define SIGMOID 3           /* sigmoid kernel type */
# define USER    4           /* user defined kernel type */

#define MAX(X,Y)    ((X)>(Y)?(X):(Y))
#define MIN(X,Y)    ((X)<(Y)?(X):(Y))
#define ABS(X)      ((X)>0?(X):(-(X)))
#define SQUARE(X)      ((X)*(X))

using namespace std;

// feature type
typedef struct feature_struct {
    int pid;
    float fval; // feature value
} feature_t;

// vector: feature/value by increasing feature num. (sparse)
typedef struct vector_struct {
    vector<feature_t> feature;
    double factor;
    double twonorm_sq;
} vect_t;

// data type (an example in data)
typedef struct data_struct {
    int outcome;    // 1 or -1
    vect_t fvec;    // feature vector
} data_t;


/** one class SVM.
 @class One_Class_SVM
 */
class One_Class_SVM {
    public:
        One_Class_SVM();
        ~One_Class_SVM();

        // load model
        virtual void load(const string model);
        virtual void load_bin(const string model);
        // for fsmo (non-lienar kernel)
        void load_support_vector(const string model);

        // save model
        virtual void save(const string model);
        virtual void save_bin(const string model);
        // for fsmo (non-lienar kernel)
        void save_support_vector(const string model);

        // show feature weight
        void show_feature();

        // remove zero feature
        void remove_zero_feature(double threshold);

        // load event and make param
        virtual void load_event(const string file);
        virtual void load_test_event(const string file);

        // get/set train_data/test_data (for cross-validation)
        vector<data_t> get_train_data() {return train_data;};
        void set_train_data(vector<data_t> new_train_data) {train_data = new_train_data;};
        void set_test_data(vector<data_t> new_test_data) {test_data = new_test_data;};

        // random_shuffle train_data
        void random_shuffle_train_data();

        // predict
        virtual int predict(ostream& f);
        // score 값을 구한다
        double eval(vect_t& vect);
        // make data
        data_t make_data(const vector<string>& cont);

        // training
        void train(string estimate);

        // clear
        void clear();

        // util
        void tokenize(const string& str, vector<string>& tokens, const string& delimiters = " ");
		void split(const string& str, vector<string>& tokens, const string& delimiter = " ");

        // 변수
        string model_file;
        int n_event;
        int n_test_event;

        // parameter
        int verbose;
        int binary;
        int skip_eval;
        int train_num;

        int buf;
        int iter;
        int period;
        double tol;

        // SVM
        int kernel_type;
        double gamma;
        double coef;
        int degree;
        int cache_size;
        int rm_inactive;
        double cost;
        double eps;
        // save support vector
        int save_sv;


    protected:	// 상속을 위해 private -> protected 
        int n_theta;				  // number of feature weights
        float *theta;  // feature weight

        map<string, int> *pred_map;
        vector<string> *pred_vec;

        vector<data_t> train_data;
        vector<data_t> test_data;

        // for SVM
        // alpha : size = const num
        vector<double> alpha;
        // alpha_history : size = const num
        vector<int> alpha_history;
        // working set : size = const num
        vector<vect_t> work_set;
        // loss : size = const num
        vector<double> loss;
        // x_norm_vec : size = const num
        vector<double> x_norm_vec;

        // sent_ids : size = const num : work_set id를 train_data id로 바꾼다
        vector<int> sent_ids;
        // work_set_ids : size = train_data.size() : train_data id->work_set id list
        vector<vector<int> > work_set_ids;
        // shirink : size = train_data.size()
        vector<int> opti;

        // sum_alpha : size = train_data.size()
        vector<double> sum_alpha;

        // slacks : size = train_data.size()
        vector<double> slacks;
        // slacks_id : size = train_data.size() : 실제 slack인 work_set id를 가리킨다
        vector<int> slacks_id;

        // cost_vec : size = const num
        vector<double> cost_vec;

        // kernel cache
        map<int, float> cache;

        // GRAM for 1-slack
        vector<vector<float> > gram;
        int gram_size;

        // for SVM
        double rho;
        double precision;

        // training - each machine learnign algorithm
        double train_one_slack_smo(int use_gram);
        double train_sgd(int pick_random_example);

        void print_start_status(string estimate);

        // fsmo_joint에 사용됨
        virtual void append_diff_vector(vector<float>& dense_vect, vect_t& vect);

        // kernel
        double kernel4gram(int vect1, int vect2);
        double kernel(int vect1, int vect2);
        double kernel(vect_t& vect1, vect_t& vect2);

        // optimize
        // 1-slack formulation
        void optimize_dual4one_slack_smo(double cost, double eps, int use_gram);

        // dot product
        double dot_product(vect_t& svect1, vect_t& svect2);

        // update weight
        void update_weight(vect_t& vect, double d);

        // length of longest vector
        double longest_vector();
};
#endif
