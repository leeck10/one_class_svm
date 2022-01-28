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

#include "one_class_svm.hpp"
#include "timer.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;


// train
void One_Class_SVM::train(string estimate) {
    print_start_status(estimate);

    if (estimate == "one_slack") {
        train_one_slack_smo(0);
    } else if (estimate == "one_slack2") {
        train_one_slack_smo(1);
    } else if (estimate == "sgd") {
        train_sgd(0);
    } else if (estimate == "sgd2") {
        train_sgd(1);
    } else {
        cerr << "!Error: " << estimate << " is not supported." << endl;
        exit(1);
    }
}

// 1-slack SMO
double One_Class_SVM::train_one_slack_smo(int use_gram) {
    size_t n = train_data.size();
    size_t niter = 1, const_num = work_set.size();
    size_t tp = 0, positive = 0, fp = 0, negative = 0;
    size_t correct = 0, argmax_count = 0;
    double ceps = 0, obj = 0, diff_obj = 0, additional_value = 0;
    double time = 0, time_qp = 0, time_viol = 0, time_psi = 0;

#ifdef _OPENMP
	cerr << "[OpenMP] Number of threads: " << omp_get_max_threads() << endl;
#endif

    // gram 초기화
    if (use_gram) {
        gram.clear();
        for (int i=0; i < gram_size; i++) {
            vector<float> gram_i;
            for (int j=0; j < gram_size; j++) {
                gram_i.push_back(-1);
            }
            gram.push_back(gram_i);
        }
    }

    // slack 초기화
    double slack = 0;
    // inactive constraint 제거
    int opt_count = 0;

    printf("iter tr-TP testTP testFP CPU-time    |w|    primal      dual const  SV slack\n");
    printf("============================================================================");
    fflush(stdout);

    if (kernel_type != LINEAR) {
        cerr << "Current version is avalable only linear kernel!" << endl;
        exit(1);
    }

    double cost1 = cost;
    double eps1 = 100;
    double old_eps = eps1;

    // dense vector
    vector<float> dense_vect(n_theta, 0);
    // c vector
    vector<int> c_vec(train_data.size(), 0);

    do { // increase precision
        timer t;
        correct = 0;
        ceps = 0.0;

        // dense vector 초기화
        for (size_t i=0; i < dense_vect.size(); i++) {
            dense_vect[i] = 0;
        }

        // slack 계산
        slack = 0;
        for (int i=0; i < (int)work_set.size(); i++) {
            double cur_cost = eval(work_set[i]);
            slack = MAX(slack, loss[i]*rho - cur_cost);
        }
        if (verbose) cerr << endl << "  slack=" << slack;

        // find violated example
        double joint_cost = 0;
        timer t_viol;
        argmax_count += train_data.size();
		#pragma omp parallel for
		for (int i=0; i < (int)train_data.size(); i++) {
			data_t& data = train_data[i];
            // evaluation
            double score = eval(data.fvec);
            c_vec[i] = 0;

            // data가 맞았는지 검사 - 맞았으면 skip
            if (score < rho) {
                c_vec[i] = 1;
			    #pragma omp atomic
                joint_cost += score;
            }
        } // example loop
        time_viol += t_viol.elapsed();

        // append_diff_vector
        timer t_psi;
        for (int i=0; i < (int)train_data.size(); i++) {
            data_t& data = train_data[i];
            if (c_vec[i] == 1) {
                append_diff_vector(dense_vect, data.fvec);
                //if (verbose) cerr << "a";
            } else {
                correct++;
            }
        }
        time_psi += t_psi.elapsed();

        double norm = 0.0;
        size_t non_empty_count = 0;
        for (int i=0; i < n_theta; i++) {
            if (dense_vect[i] != 0) {
                norm += dense_vect[i] * dense_vect[i];
                non_empty_count++;
            }
        }

        // a joint vector : linear만 고려
        vect_t joint_vect;
        joint_vect.twonorm_sq = norm;
        joint_vect.factor = 1 / double(n);
        // sparse vector로 변경한다
        for (int i=0; i < n_theta; i++) {
            if (dense_vect[i] != 0) {
                feature_t f;
                f.pid = i;
                f.fval = dense_vect[i];
                joint_vect.feature.push_back(f);
                //if (verbose) cerr << " pid=" << f.pid << ",fval=" << f.fval;
            }
        }
        if (verbose) cerr << endl << "joint_vec_twonorm_sq=" << joint_vect.twonorm_sq;

        // joint vector의 eps 계산
        //joint_cost = eval(joint_vect);
        joint_cost /= double(n);
        double joint_loss = (n - correct) / double(n);
        ceps = MAX(0, joint_loss*rho - joint_cost - slack);
        if (verbose) cerr << endl << "joint_loss=" << joint_loss*rho << " joint_cost=" << joint_cost;

        if (verbose || slack > (joint_loss*rho - joint_cost + 1e-15)) {
            if (slack > (joint_loss*rho - joint_cost + 1e-15)) {
                cerr << endl << "WARNING: Slack of most violated constraint is smaller than slack of working" << endl;
                cerr << "         set! There is probably a bug in 'find_most_violated_constraint'";
            }
            cerr << endl << "joint_loss=" << joint_loss*rho << " joint_cost=" << joint_cost;
            cerr << endl << "joint_cost(eval)=" << eval(joint_vect);
            cerr << endl << "H=" << joint_loss*rho - joint_cost << " slack=" << slack << " ceps=" << ceps << endl;
        }

        // if error, then add a joint constraint
        if (ceps > eps) {
            if (verbose) cerr << ".";
            // alpha
            alpha.push_back(0);
            // alpha_history
            alpha_history.push_back(opt_count);
            // work set
            work_set.push_back(joint_vect);
            // loss
            loss.push_back(joint_loss);
            // cost_vec
            cost_vec.push_back(joint_cost);
            // x_norm_vec
            x_norm_vec.push_back(joint_vect.factor * joint_vect.factor * joint_vect.twonorm_sq);
            const_num++;

            // set svm precision so that higher than eps of most violated const
            old_eps = eps1;
            eps1 = MIN(eps1, MAX(ceps, eps));
            if (old_eps != eps1) {
                cerr << endl << "# eps = " << eps1 << " ";
            }

            // get new QP solution
            cerr << "*";
            timer t_qp;
            optimize_dual4one_slack_smo(cost1, eps1, use_gram);
            time_qp += t_qp.elapsed();

            // inactive constraint 제거
            // rm_inactive iteration 동안 active된 적이 없는 것
            // svm-light는 50
            opt_count++;
            int remove_count = 0;
            for (int i=0; i < work_set.size() - remove_count; i++) {
                // active constraint
                if (alpha[i] > 0) {
                    alpha_history[i] = opt_count;
                }
                // rm_inactive번 안에 active 된 적이 없는 constraint
                else if (opt_count - alpha_history[i] >= rm_inactive) {
                    // 맨 뒤의 원소부터 swap한 후, remove_count만큼 맨뒤 원소들을 삭제
                    int sw_i = work_set.size() - 1 - remove_count;
                    work_set[i] = work_set[sw_i];
                    alpha[i] = alpha[sw_i];
                    alpha_history[i] = alpha_history[sw_i];
                    loss[i] = loss[sw_i];
                    cost_vec[i] = cost_vec[sw_i];
                    x_norm_vec[i] = x_norm_vec[sw_i];
                    // gram matrix
                    if (use_gram) {
                        // 1차 배열 수정
                        gram[i] = gram[sw_i];
                        // 2차 배열 수정
                        for (int j=0; j < gram_size; j++) {
                            gram[j][i] = gram[j][sw_i];
                        }
                        // 삭제되는 곳에 -1
                        for (int j=0; j < gram_size; j++) {
                            gram[sw_i][j] = -1;
                            gram[j][sw_i] = -1;
                        }
                    }
                    // cost_diff_vec
                    cost_vec[i] = cost_vec[sw_i];
                    // 나머지 변수 처리
                    i--;
                    const_num--;
                    remove_count++;
                }
            }
            if (remove_count > 0) {
                // 실제 제거
                int rm_i = work_set.size() - remove_count;
                work_set.erase(work_set.begin() + rm_i, work_set.end());
                alpha.erase(alpha.begin() + rm_i, alpha.end());
                alpha_history.erase(alpha_history.begin() + rm_i, alpha_history.end());
                loss.erase(loss.begin() + rm_i, loss.end());
                x_norm_vec.erase(x_norm_vec.begin() + rm_i, x_norm_vec.end());
                // cost_vec
                cost_vec.erase(cost_vec.begin() + rm_i, cost_vec.end());
                // test
                cerr << "r";
            }
        }

        // time
        double iter_time = t.elapsed();
        time += iter_time;

        // sv number
        int sv_num = 0;
        double sum = 0, alphasum = 0;
        for (size_t i=0; i < alpha.size(); i++) {
            if (alpha[i] > 0) {
                sum += alpha[i];
                alphasum += alpha[i] * loss[i];
                sv_num++;
            }
        }

        // obj --> model length |w| ???
        obj = 0;
        if (!skip_eval || ceps < eps) {
            if (kernel_type == LINEAR) {
                for (int i=0; i < n_theta; i++) {
                    obj += SQUARE(theta[i]);
                }
            } else {
                for (size_t i=0; i < alpha.size(); i++) {
                    if (alpha[i] > 0) {
                        for (size_t j=0; j < alpha.size(); j++) {
                            if (alpha[j] > 0) {
                                if (use_gram) {
                                    obj += alpha[i] * alpha[j] * kernel4gram(i, j);
                                } else if (kernel_type == LINEAR) {
                                    obj += alpha[i] * alpha[j] * kernel(work_set[i], work_set[j]);
                                }
                            }
                        }
                    }
                }
            }
            obj = sqrt(obj);
        }

        // continue evaluations
        double acc = 100*double(correct)/double(n_event);

        // test_data 성능
        correct = 0;
        tp = positive = fp = negative = 0;
        if (obj != 0 || diff_obj != 0) {
            vector<data_t>::iterator it = test_data.begin();
            for (; it != test_data.end(); it++) {
                data_t& data = *it;
                int y = 1;
                double score = eval(data.fvec) - rho;
                if (score < 0) y = -1;
                if (data.outcome == y) correct++;
                if (data.outcome == 1) {
                    positive++;
                    if (y == 1) tp++;
                } else {
                    negative++;
                    if (y == 1) fp++;
                }
            }
        }
        double test_acc = test_data.size() > 0 ? 100*double(correct)/double(n_test_event) : 0;
        double test_tp = test_data.size() > 0 ? 100*double(tp)/double(positive) : 0;
        double test_fp = test_data.size() > 0 ? 100*double(fp)/double(negative) : 0;

        // primal cost 계산: 0.5 * |w|^2 + C/n * L
        double primal_cost = 0, dual_cost = 0;
        if (obj != 0 || diff_obj != 0) {
            primal_cost = 0.5 * obj * obj + cost1 * (slack + ceps);
            dual_cost = alphasum - (0.5 * obj * obj);
        }

        printf("\n%3d %5.2f%% %5.2f%% %5.2f %8.2f %7.3f %9.2e %9.2e %4d %4d %6.4f", 
                niter++ , acc, test_tp, test_fp, time, obj, primal_cost, dual_cost, const_num, sv_num, slack);
        fflush(stdout);
    } while (ceps > eps);

    try {
        int const_num = 0, sv_num = 0;
        double sum = 0, alphasum = 0;
        double max_alpha = 0.0;
        const_num = alpha.size();
        for (size_t i=0; i < alpha.size(); i++) {
            if (alpha[i] != 0) {
                sum += alpha[i];
                alphasum += loss[i] * alpha[i];
                sv_num++;
            }
            max_alpha = MAX(max_alpha, alpha[i]);
        }

        slack = 0;
        for (size_t i=0; i < work_set.size(); i++) {
            slack = MAX(slack, loss[i]*rho - eval(work_set[i]));
        }

        // primal cost 계산: 0.5 * |w|^2 + C/n * L
        double primal_cost, dual_cost;
        primal_cost = 0.5 * obj * obj + cost1 * (slack + ceps);
        dual_cost = alphasum - (0.5 * obj * obj);

        cerr << endl << "Training time= " << time << endl;
        cerr << endl;
    cerr << "TPR=TP/P=TP/(TP+FN)=" << 100.0 * tp / positive << " (" << tp << "/" << positive << ")" << endl;
    cerr << "FPR=FP/N=FP/(FP+TN)=" << 100.0 * fp / negative << " (" << fp << "/" << negative << ")" << endl;
    cerr << "TP=" << tp << " FN=" << positive-tp << " FP=" << fp << " TN=" << negative-fp <<  endl;

        cerr << endl << "Final epsilon on KKT-Conditions: " << ceps << endl;
        cerr << "const=" << const_num << " SV=" << sv_num << " alphasum=" << alphasum << " sum(a)=" << sum << " max(a)=" << max_alpha << endl;
        cerr << "slack=" << slack << " rho=" << rho << endl;
        cerr << "|w|=" << obj << endl;
        cerr << "primal_cost(upper bound)=" << primal_cost << endl;
        cerr << "dual object=" << dual_cost << endl;
        cerr << "duality gap=" << primal_cost - dual_cost << endl;
        cerr << "longest |x|=" << longest_vector() << endl;
		cerr << "Runtime(sec): QP=" << time_qp << " Argmax=" << time_viol << " psi=" << time_psi << endl;
        cerr << "Runtime(%): QP=" << 100*time_qp/time << " Argmax=" << 100*time_viol/time;
        cerr << " psi=" << 100*time_psi/time << " others=" << 100*(time-time_qp-time_viol-time_psi)/time << endl;
        cerr << "Number of calls to 'find_most_violated_constraint': " << argmax_count << endl;
    } catch (std::exception& e) {
        cerr << endl << "std::exception caught:" << e.what() << endl;
    }

    // sv 저장
    if (kernel_type != LINEAR || save_sv) {
        string sv_file = model_file + ".sv";
        cerr << "support vector saving to " << sv_file << " ... ";
        save_support_vector(sv_file);
        cerr << "done." << endl;
    }

    return time;
}


// Stochastic Gradient Descent (primal optimization)
// f = 0.5 * lambda * |w|^2 + (1/n)sum{L(x;w)}
double One_Class_SVM::train_sgd(int pick_random_example) {
    int i, j, k;
    int niter = 1, weight_num = 0;
    int tp = 0, positive = 0, fp = 0, negative = 0;
    int correct = 0, total = 0, argmax_count = 0;
    double time = 0, time_qp = 0, time_viol = 0, time_psi = 0;

#ifdef _OPENMP
	cerr << "[OpenMP] Number of threads: " << omp_get_max_threads() << endl;
#endif

    printf("iter primal_cost      |w|   d(cost) trainTP test-TP test-FP CPU-time\n");
    printf("====================================================================");
    fflush(stdout);

    if (kernel_type != LINEAR) {
        cerr << "Current version is avalable only linear kernel!" << endl;
        exit(1);
    }

    // C/n
    double n = (double) train_data.size();
    // lambda = 1/C
    double lambda = 1.0 / cost;

    double f = 0.0, old_f = 0.0;
    double wscale = 1, old_wscale = 1;
    double obj = 0, diff_obj = 0;
    double dcost = 1;
    double t_i = 1;
    double test_acc = 0;

    // shuffle
    vector<int> train_data_index;
    if (!pick_random_example) {
        for (i=0; i < train_data.size(); i++)
            train_data_index.push_back(i);
    }

    for (; niter <= iter; niter++) {
        timer t;
        total = 0;
        correct = 0;
        old_f = f;
        f = 0;

        // shuffle
        if (!pick_random_example) {
            cerr << "r";
            random_shuffle(train_data_index.begin(), train_data_index.end());
            cerr << ".";
        }

        for (size_t data_i = 0; data_i < train_data.size();) {
			#pragma omp parallel for private(j)
            for (i = 0; i < buf; i++) {
				int skip = 0;
				#pragma omp critical (data_i)
				if (data_i++ >= train_data.size()) skip = 1;
				if (skip) continue;
                // choose random example
				int r;
                if (pick_random_example) {
                    r = ((int)rand()) % train_data.size();
                } else {
					// shuffle - 다시 한번 체크해 줌
					if (data_i-1 >= train_data_index.size()) continue;
                    r = train_data_index[data_i-1];
                }
                data_t& data = train_data[r];
    
                // find most violated contraint
				#pragma omp atomic
                argmax_count++;

                timer t_viol;
                // evaluation
                double score = eval(data.fvec);
                // wscale 반영
                score *= wscale;
                // loss = max{0, rho - score} 계산
                double cur_loss = MAX(0, rho - score);

				#pragma omp atomic
                time_viol += t_viol.elapsed();
    
                // data가 맞았는지 검사
				if (score >= rho) {
					#pragma omp atomic
					correct++;
				}
    
                timer t_psi;
				#pragma omp critical (work_set)
                {
                    // work set
                    work_set.push_back(data.fvec);
                    // loss
                    loss.push_back(cur_loss);
                    // f: hinge loss
                    f += cur_loss / n;
                }
				#pragma omp atomic
                time_psi += t_psi.elapsed();
            }
    
            // Stochastic Gradient Decent
            timer t_qp;
            // pegasos
            double eta = 1.0 / (lambda * (1 + t_i));
            // check eta
            if (eta * lambda > 0.9) {
                cerr << "e";
                eta = 0.9 / lambda;
            }
    
            double s = 1 - eta * lambda;
            if (s > 0) {
                old_wscale = wscale;
                wscale *= s;
            }
    
            // update w
            for (i=0; i < work_set.size(); i++) {
                vect_t& vect = work_set[i];
                double cur_loss = loss[i];
                if (cur_loss > 0) {
                    // hinge loss: g = lambda*w - (1/n)sum{X_i}
                    double factor = (1.0 / wscale) * eta * vect.factor / double(work_set.size());
                    for (j=0; j < vect.feature.size(); j++) {
                        int fid = vect.feature[j].pid;
                        double val = vect.feature[j].fval;
                        double old_theta = theta[fid];
                        // update theta
                        theta[fid] += factor * val;
                        // update obj : obj는 scale을 뺀 값을 저장
                        obj -= old_theta * old_theta;
                        obj += theta[fid] * theta[fid];
                    }
                }
            }

            // scaling
            if (wscale < 1e-7) {
                cerr << "s";
                //cerr << endl << "wscale=" << wscale;
                for (int i=0; i < n_theta; i++) {
                    if (theta[i] != 0) theta[i] *= wscale;
                }
                obj *= wscale * wscale;
                wscale = 1;
                cerr << ".";
            }

            work_set.clear();
            loss.clear();
            t_i++;
            time_qp += t_qp.elapsed();
        }

        double iter_time = t.elapsed();
        time += iter_time;

        // f
        f += 0.5 * lambda * (wscale*wscale*obj);
        dcost = (dcost + ABS(old_f - f)/MAX(old_f,f)) / 2.0;

        // continue evaluations
        double acc = correct/n;

        // test_data 성능
        correct = 0;
        tp = positive = fp = negative = 0;
        if (!skip_eval || (period == 0 && niter % 10 == 0) || (period > 0 && niter % period == 0) || dcost < tol || niter == iter) {
            // scaling
            if (wscale < 1) {
                cerr << "s";
                for (int i=0; i < n_theta; i++) {
                    if (theta[i] != 0) theta[i] *= wscale;
                }
                obj *= wscale * wscale;
                wscale = 1;
            }

            for (int i = 0; i < test_data.size(); i++) {
                data_t& data = test_data[i];
                int y = 1;
                double score = eval(data.fvec) - rho;
                if (score < 0) y = -1;
                if (data.outcome == y) correct++;
                if (data.outcome == 1) {
                    positive++;
                    if (y == 1) tp++;
                } else {
                    negative++;
                    if (y == 1) fp++;
                }
            }
        }
        test_acc = test_data.size() > 0 ? 100*double(correct)/double(n_test_event) : 0;
        double test_tp = test_data.size() > 0 ? 100*double(tp)/double(positive) : 0;
        double test_fp = test_data.size() > 0 ? 100*double(fp)/double(negative) : 0;

        printf("\n%4d  %.3e %8.3f %9.6f %6.2f%% %6.2f%% %6.2f%% %7.2f ", niter,
            cost*f, sqrt(wscale*wscale*obj), dcost, (acc*100), test_tp, test_fp, time);

        fflush(stdout);

        // 끝
        if (dcost < tol) {
            printf("\nTraining terminats succesfully in %.2f seconds\n", time);
            break;
        }

        // 중간중간 저장 -- by leeck
        if (period > 0 && niter < iter && (niter % period == 0)) {
            // scaling
            if (wscale < 1) {
                cerr << "s";
                for (int i=0; i < n_theta; i++) {
                    if (theta[i] != 0) theta[i] *= wscale;
                }
                obj *= wscale * wscale;
                wscale = 1;
            }

            char temp[200];
            if (model_file != "") {
                cerr << "model saving to " << model_file << "." << niter << " ... ";
            }

            if (model_file != "") {
                sprintf(temp, "%s.%d", model_file.c_str(), niter);
                save(string(temp));
                cerr << "done." << endl;
            }
        } // end of train_data
    }

    // scaling
    if (wscale < 1) {
        cerr << "s";
        for (int i=0; i < n_theta; i++) {
            if (theta[i] != 0) theta[i] *= wscale;
        }
        obj *= wscale * wscale;
        wscale = 1;
    }

    if (niter > iter) {
        printf("\nMaximum numbers of %d iterations reached in %.2f seconds\n", iter, time);
    }

    cerr << endl;
    cerr << "TPR=TP/P=TP/(TP+FN)=" << 100.0 * tp / positive << " (" << tp << "/" << positive << ")" << endl;
    cerr << "FPR=FP/N=FP/(FP+TN)=" << 100.0 * fp / negative << " (" << fp << "/" << negative << ")" << endl;
    cerr << "TP=" << tp << " FN=" << positive-tp << " FP=" << fp << " TN=" << negative-fp <<  endl;
    cerr << "|w|=" << sqrt(obj) << endl;
    cerr << "Runtime(%): SGD=" << 100*time_qp/time << " Argmax=" << 100*time_viol/time;
    cerr << " psi=" << 100*time_psi/time << " others=" << 100*(time-time_qp-time_viol-time_psi)/time << endl;
    cerr << "Number of calls to 'find_most_violated_constraint': " << argmax_count << endl;

    return time;
}


// print start status
void One_Class_SVM::print_start_status(string estimate) {
    printf("\nStarting %s iterations...\n", estimate.c_str());
    printf("Number of Data : %d\n", train_data.size());
    printf("Number of Predicates: %d\n", (*pred_vec).size());
    printf("Number of Parameters: %d\n", n_theta);
    printf("[SVM] Cost:%g C/n:%g eps:%g buf:%d rm_inactive:%d\n", cost,
            cost/double(train_data.size()), eps, buf, rm_inactive);
    printf("[SVM] kernel:%d gamma:%g degree:%d coef:%g\n", kernel_type, gamma, degree, coef);
    fflush(stdout);
}


// kernel function for joint constraint
double One_Class_SVM::kernel4gram(int v1, int v2) {
    int i, j;
    int max = MAX(v1,v2);

    if (max >= gram_size) {
        cerr << endl << "GRAM size is small: " << gram_size << " , extended:" << 2*gram_size << " ... ";
        // 기존 항목 size 늘임
        for (i=0; i < gram_size; i++) {
            for (j=gram_size; j < 2*gram_size; j++) {
                gram[i].push_back(-1);
            }
        }
        // 추가
        for (i=gram_size; i < 2*gram_size; i++) {
            vector<float> gram_i;
            for (j=0; j < 2*gram_size; j++) {
                gram_i.push_back(-1);
            }
            gram.push_back(gram_i);
        }
        gram_size *= 2;
        cerr << "Done." << endl;
    }

    float result = gram[v1][v2];

    if (result == -1) {
        float val;
        if (kernel_type == LINEAR || cache_size == 0) {
            val = kernel(work_set[v1], work_set[v2]);
        } else {
            val = kernel(v1, v2);
        }
        gram[v1][v2] = val;
        gram[v2][v1] = val;
        result = val;
    }
    return result;
}


// optimize dual for 1-slack smo: shared_slack
// non-bound와 bound를 분리해서 실행
void One_Class_SVM::optimize_dual4one_slack_smo(double cost, double eps1, int use_gram) {
    timer t;
    int changed_num = 0, qiter = 0, i, j, k;
    int smo_count = 0;

    // shrink: svm-light와 비슷(100): qiter - lastiter > 100
    vector<int> shrink(work_set.size(), 0);
    vector<int> lastiter(work_set.size(), 0);
    int shrink_round = 1, shrink_count = 0;
    
    // cost_diff 계산 : for non-linear : rho 값은 제외하고 저장 함
    if (kernel_type != LINEAR || use_gram) {
        for (i = (int)cost_vec.size(); i < (int)work_set.size(); i++) {
            double cur_cost;
            cur_cost = eval(work_set[i]);
            cost_vec.push_back(cur_cost);
        }
    }

    // diff_alpha
    vector<double> diff_alpha(alpha.size(), 0);

    // sum_alpha 계산 : sum(alpha) <= C
    double sum_alpha = 0;
    for (i=0; i < (int)work_set.size(); i++) {
        sum_alpha += alpha[i];
    }

    do {
        changed_num = 0;

        // non-bound인 경우
        if (sum_alpha < cost - precision) {
            // non-bound인 경우 (fsmo 적용)
            if (verbose) cerr << " fsmo";
            for (i=0; i < (int)work_set.size(); i++) {
                // shrink, final_opt_check 발동시에는 shrink 안함
                if (shrink[i] == shrink_round) {
                    shrink_count++;
                    if (verbose) cerr << "s";
                    continue;
                }

                double x_norm = x_norm_vec[i];
                if (x_norm == 0) continue;

                double H_x, cur_cost;
                if (kernel_type == LINEAR && !use_gram) {
                    cur_cost = eval(work_set[i]);
                } else {
                    cur_cost = cost_vec[i];
                }
                H_x = loss[i]*rho - cur_cost;
                if (verbose) cerr << " H=" << H_x;

                // KKT condition check : 0 <= sum(a[i]) <= C
                // alpha 값이 증가하는 것만 허용 (빨리 sum(a[i]) == C 가 되도록)
                //if ((H_x > 0.5*eps1 && alpha[i] < cost - precision)) {}
                if ((H_x > 0.5*eps1 && alpha[i] < cost - precision) || (H_x < -0.5*eps1 && alpha[i] > precision)) {
                    // margin 구하기
                    double org_margin = H_x / x_norm;
                    //if (verbose) cerr << " org_margin=" << org_margin;

                    // lower bound : sum(alpha) is clipped to the [0,C]
                    double L = -alpha[i];
                    // upper bound : sum(alpha) is clipped to the [0,C]
                    double H = cost - sum_alpha;

                    double margin = MAX(L, MIN(H, org_margin));
                    if (verbose) cerr << " margin=" << margin;

                    // w 업데이트
                    if (margin > precision || margin < -precision) {
                        if (!use_gram) {
                            update_weight(work_set[i], margin);
                        }
                        // alpha 업데이트
                        alpha[i] += margin;
                        diff_alpha[i] += margin;
                        sum_alpha += margin;

                        // cost_vec 업데이트 : for non-linear kernel
                        if (kernel_type != LINEAR || use_gram) {
                            for (j=0; j < (int)work_set.size(); j++) {
                                double prod;
                                if (use_gram) {
                                    prod = kernel4gram(j, i);
                                } else {
                                    prod = kernel(work_set[j], work_set[i]);
                                }
                                cost_vec[j] += margin * prod;
                            }
                        }

                        changed_num++;
                        // for shrink
                        lastiter[i] = qiter;
                    }
                    if (verbose) cerr << " sum_alpha=" << sum_alpha;
                }
            }
        }
        // bound 되었을 경우, working set selection (SMO 알고리즘 적용)
        else {
            double g_max = -1e10, g_min = 1e10, obj_min = 1e10;
            int max_i = -1, min_j = -1;

            // i 선택
            for (i=0; i < (int)work_set.size(); i++) {
                if (x_norm_vec[i] == 0) continue;
                if (alpha[i] > cost - precision) continue;

                double H_x, cur_cost;
                if (use_gram) {
                    cur_cost = cost_vec[i];
                } else {
                    cur_cost = eval(work_set[i]);
                }
                H_x = loss[i]*rho - cur_cost;
                if (H_x > g_max) {
                    g_max = H_x;
                    max_i = i;
                }
            }
            if (max_i < 0) continue;
            i = max_i;

            // second order - j 선택
            double min_H_x = 0;
            for (j=0; j < (int)work_set.size(); j++) {
                if (j != i && alpha[j] > precision) {
                    double cur_H_x;
                    if (use_gram) {
                        cur_H_x = loss[j]*rho - cost_vec[j];
                    } else {
                        cur_H_x = loss[j]*rho - eval(work_set[j]);
                    }
                    // g_min
                    g_min = MIN(g_min, cur_H_x);
                    // second order
                    double prod_ij;
                    if (use_gram) {
                        prod_ij = kernel4gram(i, j);
                    } else {
                        prod_ij = kernel(work_set[i], work_set[j]);
                    }
                    double x_norm = x_norm_vec[i] + x_norm_vec[j] - 2*prod_ij;
                    double cur_obj = -SQUARE(g_max - cur_H_x) / x_norm;
                    if (cur_obj < obj_min) {
                        obj_min = cur_obj;
                        min_H_x = cur_H_x;
                        min_j = j;
                    }
                }
            }

            if (min_j < 0) continue;
            if (g_max - g_min < eps1) continue;
            j = min_j;

            double prod_ij;
            if (use_gram) {
                prod_ij = kernel4gram(i, j);
            } else {
                prod_ij = kernel(work_set[i], work_set[j]);
            }
            double x_norm = x_norm_vec[i] + x_norm_vec[j] - 2*prod_ij;
            // a_j margin
            // first order
            //double margin = (g_max - g_min) / x_norm;
            // second order
            double margin = (g_max - min_H_x) / x_norm;
            margin = MIN(alpha[j], margin);

            if (margin > precision || margin < -precision) {
                if (!use_gram) {
                    update_weight(work_set[i], margin);
                    update_weight(work_set[j], -margin);
                }
                if (verbose) cerr << " smo_margin=" << margin;
                // alpha 업데이트
                alpha[i] += margin;
                alpha[j] -= margin;
                diff_alpha[i] += margin;
                diff_alpha[j] -= margin;

                // cost_diff_vec 업데이트 : for non-linear kernel
                if (kernel_type != LINEAR || use_gram) {
                    for (k=0; k < (int)work_set.size(); k++) {
                        double prod_i, prod_j;
                        if (use_gram) {
                            prod_i = kernel4gram(k, i);
                            prod_j = kernel4gram(k, j);
                        } else {
                            prod_i = kernel(work_set[k], work_set[i]);
                            prod_j = kernel(work_set[k], work_set[j]);
                        }
                        cost_vec[k] += margin * prod_i;
                        cost_vec[k] -= margin * prod_j;
                    }
                }

                smo_count++;
                changed_num++;
                // for shrink
                lastiter[i] = qiter;
                lastiter[j] = qiter;
            }
        }
        qiter++;

        if (changed_num == 0 && shrink_count > 0) {
            cerr << "F";
            changed_num = 1;
        }
    } while (qiter < 100000 && changed_num > 0);

    // diff_alpha 값 w에 반영
    if (use_gram) {
        for (i=0; i < (int)diff_alpha.size(); i++) {
            if (diff_alpha[i] != 0)
                update_weight(work_set[i], diff_alpha[i]);
        }
    }

    // shrink 정보
    if (shrink_count > 0) cerr << "s";

    if (smo_count > 0) cerr << "+" << smo_count;

    // 걸린 시간
    if (qiter >= 10000 || t.elapsed() >= 1000) {
        cerr << "(" << qiter << ":" << t.elapsed() << ")";
    }
}


// update weight vector
void One_Class_SVM::update_weight(vect_t& vect, double d) {
    for (int i=0; i < vect.feature.size(); i++) {
        double factor = d * vect.factor;
        int fid = vect.feature[i].pid;
        theta[fid] += factor * vect.feature[i].fval;
    }
}


// append_diff_vector : s_vect에 {f(xi,yi) - f(xi,y)}을 더함
// s_vect는 dense vector를 사용함
void One_Class_SVM::append_diff_vector(vector<float>& dense_vect, vect_t& vect) {
    // non-linear kernel
    if (kernel_type != LINEAR) {
        cerr << "append_diff_vector : linear kernel is available!" << endl;
        exit(1);
    }
    // sparse vector
    if (vect.factor == 1) {
        for (size_t i=0; i < vect.feature.size(); i++) {
            int fid = vect.feature[i].pid;
            dense_vect[fid] += vect.feature[i].fval;
        }
    } else {
        for (size_t i=0; i < vect.feature.size(); i++) {
            int fid = vect.feature[i].pid;
            dense_vect[fid] += vect.factor * vect.feature[i].fval;
        }
    }
}

// length of longest vector
double One_Class_SVM::longest_vector() {
    double max_len = 0, len = 0;
    for (size_t i=0; i < alpha.size(); i++) {
        len = sqrt(kernel(work_set[i], work_set[i]));
        if (len > max_len) max_len = len;
    }
    return max_len;
}



