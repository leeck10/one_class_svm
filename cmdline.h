/* cmdline.h */

/* File autogenerated by gengetopt version 2.20  */

#ifndef CMDLINE_H
#define CMDLINE_H

/* If we use autoconf.  */
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#ifndef CMDLINE_PARSER_PACKAGE
#define CMDLINE_PARSER_PACKAGE "one_class_svm_tool"
#endif

#ifndef CMDLINE_PARSER_VERSION
#define CMDLINE_PARSER_VERSION "2012.5"
#endif

struct gengetopt_args_info
{
  const char *help_help; /* Print help and exit help description.  */
  const char *version_help; /* Print version and exit help description.  */
  float cost_arg;	/* set Cost of one class SVM (default='1').  */
  char * cost_orig;	/* set Cost of one class SVM original value given at command line.  */
  const char *cost_help; /* set Cost of one class SVM help description.  */
  char * model_arg;	/* set model file name.  */
  char * model_orig;	/* set model file name original value given at command line.  */
  const char *model_help; /* set model file name help description.  */
  int binary_flag;	/* save/load model in binary format (default=off).  */
  const char *binary_help; /* save/load model in binary format help description.  */
  int skip_eval_flag;	/* skip test set evaluation in the middle of training (default=off).  */
  const char *skip_eval_help; /* skip test set evaluation in the middle of training help description.  */
  int random_arg;	/* use random_shuffle in train_data (disabled if use 0) (default='0').  */
  char * random_orig;	/* use random_shuffle in train_data (disabled if use 0) original value given at command line.  */
  const char *random_help; /* use random_shuffle in train_data (disabled if use 0) help description.  */
  int train_num_arg;	/* set number of sentence in train_data for training (for experiments) (disabled if use 0) (default='0').  */
  char * train_num_orig;	/* set number of sentence in train_data for training (for experiments) (disabled if use 0) original value given at command line.  */
  const char *train_num_help; /* set number of sentence in train_data for training (for experiments) (disabled if use 0) help description.  */
  int iter_arg;	/* iterations for training algorithm (sgd) (default='100').  */
  char * iter_orig;	/* iterations for training algorithm (sgd) original value given at command line.  */
  const char *iter_help; /* iterations for training algorithm (sgd) help description.  */
  float tol_arg;	/* set Tolerance (default='1e-04').  */
  char * tol_orig;	/* set Tolerance original value given at command line.  */
  const char *tol_help; /* set Tolerance help description.  */
  int period_arg;	/* save model periodically (sgd) (default='0').  */
  char * period_orig;	/* save model periodically (sgd) original value given at command line.  */
  const char *period_help; /* save model periodically (sgd) help description.  */
  int verbose_flag;	/* verbose mode (default=off).  */
  const char *verbose_help; /* verbose mode help description.  */
  char * output_arg;	/* prediction output filename.  */
  char * output_orig;	/* prediction output filename original value given at command line.  */
  const char *output_help; /* prediction output filename help description.  */
  float epsilon_arg;	/* set epsilon (default='0.001').  */
  char * epsilon_orig;	/* set epsilon original value given at command line.  */
  const char *epsilon_help; /* set epsilon help description.  */
  int buf_arg;	/* number of new constraints to accumulated before recomputing the QP (sgd) (default='10').  */
  char * buf_orig;	/* number of new constraints to accumulated before recomputing the QP (sgd) original value given at command line.  */
  const char *buf_help; /* number of new constraints to accumulated before recomputing the QP (sgd) help description.  */
  int rm_inactive_arg;	/* inactive constraints are removed (iteration) (smo, one_slack_smo) (default='50').  */
  char * rm_inactive_orig;	/* inactive constraints are removed (iteration) (smo, one_slack_smo) original value given at command line.  */
  const char *rm_inactive_help; /* inactive constraints are removed (iteration) (smo, one_slack_smo) help description.  */
  int save_sv_flag;	/* save support vector (one_slack) (default=off).  */
  const char *save_sv_help; /* save support vector (one_slack) help description.  */
  int kernel_arg;	/* kernel type (0=Linear 1=POLY 2=RBF 3=SIGMOID in SVMs) (default='0').  */
  char * kernel_orig;	/* kernel type (0=Linear 1=POLY 2=RBF 3=SIGMOID in SVMs) original value given at command line.  */
  const char *kernel_help; /* kernel type (0=Linear 1=POLY 2=RBF 3=SIGMOID in SVMs) help description.  */
  float gamma_arg;	/* gamma in RBF kernel (in SVMs) (default='1').  */
  char * gamma_orig;	/* gamma in RBF kernel (in SVMs) original value given at command line.  */
  const char *gamma_help; /* gamma in RBF kernel (in SVMs) help description.  */
  float coef_arg;	/* coef in POLY/SIGMOID kernel (in SVMs) (default='1').  */
  char * coef_orig;	/* coef in POLY/SIGMOID kernel (in SVMs) original value given at command line.  */
  const char *coef_help; /* coef in POLY/SIGMOID kernel (in SVMs) help description.  */
  int degree_arg;	/* degree in POLY kernel (in SVMs) (default='3').  */
  char * degree_orig;	/* degree in POLY kernel (in SVMs) original value given at command line.  */
  const char *degree_help; /* degree in POLY kernel (in SVMs) help description.  */
  const char *predict_help; /* prediction mode, default is training mode help description.  */
  const char *show_help; /* show-feature mode help description.  */
  const char *convert_help; /* convert mode ('txt model to bin model' or 'bin model to txt model (with -b)') and remove zero features (with --tol threshold) help description.  */
  const char *one_slack_help; /* use 1-slack one class SVM without Gram matrix help description.  */
  const char *one_slack2_help; /* use 1-slack one class SVM using Gram matrix help description.  */
  const char *sgd_help; /* use SGD (Pegasos) in primal optimization (random shuffled train_data) help description.  */
  const char *sgd2_help; /* use Stachastic Gradient Descent (Pegasos) in primal optimization (pick random examples) help description.  */
  
  int help_given ;	/* Whether help was given.  */
  int version_given ;	/* Whether version was given.  */
  int cost_given ;	/* Whether cost was given.  */
  int model_given ;	/* Whether model was given.  */
  int binary_given ;	/* Whether binary was given.  */
  int skip_eval_given ;	/* Whether skip_eval was given.  */
  int random_given ;	/* Whether random was given.  */
  int train_num_given ;	/* Whether train_num was given.  */
  int iter_given ;	/* Whether iter was given.  */
  int tol_given ;	/* Whether tol was given.  */
  int period_given ;	/* Whether period was given.  */
  int verbose_given ;	/* Whether verbose was given.  */
  int output_given ;	/* Whether output was given.  */
  int epsilon_given ;	/* Whether epsilon was given.  */
  int buf_given ;	/* Whether buf was given.  */
  int rm_inactive_given ;	/* Whether rm_inactive was given.  */
  int save_sv_given ;	/* Whether save_sv was given.  */
  int kernel_given ;	/* Whether kernel was given.  */
  int gamma_given ;	/* Whether gamma was given.  */
  int coef_given ;	/* Whether coef was given.  */
  int degree_given ;	/* Whether degree was given.  */
  int predict_given ;	/* Whether predict was given.  */
  int show_given ;	/* Whether show was given.  */
  int convert_given ;	/* Whether convert was given.  */
  int one_slack_given ;	/* Whether one_slack was given.  */
  int one_slack2_given ;	/* Whether one_slack2 was given.  */
  int sgd_given ;	/* Whether sgd was given.  */
  int sgd2_given ;	/* Whether sgd2 was given.  */

  char **inputs ; /* unamed options */
  unsigned inputs_num ; /* unamed options number */
  int MODE_group_counter; /* counter for group MODE */
  int Parameter_Estimate_Method_group_counter; /* counter for group Parameter_Estimate_Method */
} ;

extern const char *gengetopt_args_info_purpose;
extern const char *gengetopt_args_info_usage;
extern const char *gengetopt_args_info_help[];

int cmdline_parser (int argc, char * const *argv,
  struct gengetopt_args_info *args_info);
int cmdline_parser2 (int argc, char * const *argv,
  struct gengetopt_args_info *args_info,
  int override, int initialize, int check_required);
int cmdline_parser_file_save(const char *filename,
  struct gengetopt_args_info *args_info);

void cmdline_parser_print_help(void);
void cmdline_parser_print_version(void);

void cmdline_parser_init (struct gengetopt_args_info *args_info);
void cmdline_parser_free (struct gengetopt_args_info *args_info);

int cmdline_parser_required (struct gengetopt_args_info *args_info,
  const char *prog_name);


#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* CMDLINE_H */
