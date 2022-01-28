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

#include <cassert>

#if HAVE_GETTIMEOFDAY
    #include <sys/time.h> // for gettimeofday()
#endif

#include <cstdlib>
#include <cassert>
#include <stdexcept> //for std::runtime_error
#include <memory>    //for std::bad_alloc
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "cmdline.h"

#include "one_class_svm.hpp"

using namespace std;

int main(int argc,char* argv[]) {
    One_Class_SVM *osvm;
    string model_file;

    // time check
    struct tm *t;
    time_t timer;
    time(&timer);
    t = localtime(&timer);
    int year = t->tm_year + 1900;
    int month = t->tm_mon + 1;
    int day = t->tm_mday;
    //printf("%d/%d/%d\n", year, month, day);
    if (year >= 2020 || year <= 2010) {
        printf("%s\n", "본 프로그램의 사용기간이 만료 되었습니다.");
        printf("%s\n", "leeck@kangwon.ac.kr or leeck10@gmail.com (이창기, 010-3308-0337)로 문의해주시기 바랍니다.");
        exit(1);
    }

    try {
        gengetopt_args_info args_info;

        /* let's call our CMDLINE Parser */
        if (cmdline_parser (argc, argv, &args_info) != 0)
            return EXIT_FAILURE;

        string in_file;
        string out_file;
        string test_file;
        int random = 0;

        osvm = new One_Class_SVM();

        // model
        if (args_info.model_given) {
            model_file = args_info.model_arg;
            osvm->model_file = model_file;
        }
        // output
        if (args_info.output_given) {
            out_file = args_info.output_arg;
        }
        // kernel
        if (args_info.kernel_given) {
            osvm->kernel_type = args_info.kernel_arg;
        }
        // gamma
        if (args_info.gamma_given) {
            osvm->gamma = args_info.gamma_arg;
        }
        // coef
        if (args_info.coef_given) {
            osvm->coef = args_info.coef_arg;
        }
        // degree
        if (args_info.degree_given) {
            osvm->degree = args_info.degree_arg;
        }
        // rm_inactive
        if (args_info.rm_inactive_given) {
            osvm->rm_inactive = args_info.rm_inactive_arg;
        }
        // buf
        if (args_info.buf_given) {
            osvm->buf = args_info.buf_arg;
        }
        // iter
        if (args_info.iter_given) {
            osvm->iter = args_info.iter_arg;
        }
        // period
        if (args_info.period_given) {
            osvm->period = args_info.period_arg;
        }
        // cost
        if (args_info.cost_given) {
            osvm->cost = args_info.cost_arg;
        }
        // tol
        if (args_info.tol_given) {
            osvm->tol = args_info.tol_arg;
        }
        // eps
        if (args_info.epsilon_given) {
            osvm->eps = args_info.epsilon_arg;
        }
        // random
        if (args_info.random_given) {
            random = args_info.random_arg;
        }
        // train_num
        if (args_info.train_num_given) {
            osvm->train_num = args_info.train_num_arg;
        }

        osvm->verbose = args_info.verbose_flag;
        osvm->binary = args_info.binary_flag;
        osvm->skip_eval = args_info.skip_eval_flag;

        if (args_info.inputs_num > 0) {
            in_file = args_info.inputs[0];
            if (args_info.inputs_num > 1) {
                test_file = args_info.inputs[1];
            }
        } else if (!args_info.show_given && 
                !args_info.convert_given) {
            cmdline_parser_print_help();
            return EXIT_FAILURE;
        }

        string estimate = "one_slack";
        if (args_info.one_slack2_given)
            estimate = "one_slack2";
        if (args_info.sgd_given)
            estimate = "sgd";
        if (args_info.sgd2_given)
            estimate = "sgd2";

		if (args_info.predict_given) {
			// predict mode
			if (model_file == "")
				throw runtime_error("model name not given");

			if (osvm->binary) osvm->load_bin(model_file);
            else osvm->load(model_file);

			cerr << "Loading predicting events from " << in_file << endl;
            osvm->load_test_event(in_file);

            // ostream
            ofstream f(out_file.c_str());
            if (f) osvm->predict(f);
            else osvm->predict(cout);
		} else if (args_info.show_given) {
			// show-feature mode
			if (model_file == "")
				throw runtime_error("model name not given");

			if (osvm->binary) osvm->load_bin(model_file);
            else osvm->load(model_file);

			osvm->show_feature();
        } else if (args_info.convert_given) {
			// convert mode: txt to bin or bin to txt (with -b)
			if (model_file == "")
				throw runtime_error("model name not given");

            if (osvm->binary) osvm->load_bin(model_file);
            else osvm->load(model_file);
            osvm->remove_zero_feature(osvm->tol);
			if (osvm->binary) osvm->save(model_file + ".txt");
            else osvm->save_bin(model_file + ".bin");
		} else {
			// training mode
			printf("\nLoading training events from %s\n", in_file.c_str());
            osvm->load_event(in_file);
            cerr << "load_event is done!" << endl;

            if (random != 0) {
                srand(random);
                osvm->random_shuffle_train_data();
                cerr << "random_shuffle is done!" << endl;
            }

            // 학습하면서 test
			if (test_file != "") {
			    printf("\nLoading testing events from %s\n", test_file.c_str());
				osvm->load_test_event(test_file);
			}

			osvm->train(estimate);

			if (model_file != "") {
				cerr << "model saving ... ";
				if (osvm->binary) osvm->save_bin(model_file);
                else osvm->save(model_file);
				cerr << "done." << endl;
			} else {
				cerr << "Warning: model name not given, no model saved" << endl;
			}
		}
    } catch (std::bad_alloc& e) {
        cerr << "std::bad_alloc caught: out of memory" << e.what() << endl;
        return EXIT_FAILURE;
    } catch (std::runtime_error& e) {
        cerr << "std::runtime_error caught:" << e.what() << endl;
        return EXIT_FAILURE;
    } catch (std::exception& e) {
        cerr << "std::exception caught:" << e.what() << endl;
        return EXIT_FAILURE;
    } catch (...) {
        cerr << "unknown exception caught!" << endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

