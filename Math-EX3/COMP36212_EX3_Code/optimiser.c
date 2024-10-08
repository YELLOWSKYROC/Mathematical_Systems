#include "optimiser.h"
#include "mnist_helper.h"
#include "neural_network.h"
#include "math.h"
#include <time.h>

// Function declarations
void update_parameters(unsigned int batch_size);
void update_parameters_momentum(unsigned int batch_size, unsigned int epoch_counter);
void update_parameters_adam(unsigned int batch_size);

// Optimisation parameters
unsigned int log_freq = 30000; // Compute and print accuracy every log_freq iterations

// Paramters passed from command line arguments
unsigned int num_batches;
unsigned int batch_size;
unsigned int total_epochs;
double learning_rate;
double alpha;
double initial_learning_rate;
double final_learning_rate;
int mode;

void print_training_stats(unsigned int epoch_counter, unsigned int total_iter, double learning_rate, double mean_loss, double test_accuracy){
    printf("Epoch: %u,  Total iter: %u,  Lr: %f, Mean Loss: %0.12f,  Test Acc: %f\n", epoch_counter+1, total_iter, learning_rate, mean_loss, test_accuracy);
}

void initialise_optimiser(double cmd_line_learning_rate, int cmd_line_batch_size, int cmd_line_total_epochs, double cmd_alpha, double cmd_initial_learning_rate, double cmd_final_learning_rate, int cmd_mode){
    batch_size = cmd_line_batch_size;
    learning_rate = cmd_line_learning_rate;
    total_epochs = cmd_line_total_epochs;

    alpha = cmd_alpha;
    initial_learning_rate = cmd_line_learning_rate;
    final_learning_rate = cmd_final_learning_rate;
    mode = cmd_mode;
    
    num_batches = total_epochs * (N_TRAINING_SET / batch_size);
    printf("Optimising with paramters: \n\tepochs = %u \n\tbatch_size = %u \n\tnum_batches = %u\n\tlearning_rate = %f\n\tinitial_learning_rate = %f\n\tfinal_learning_rate = %f\n\tmode = %u\n\talpha = %f\n\n",
           total_epochs, batch_size, num_batches, learning_rate, initial_learning_rate, final_learning_rate, mode, alpha);
}

void run_optimisation(void){
    unsigned int training_sample = 0;
    unsigned int total_iter = 0;
    double obj_func = 0.0;
    unsigned int epoch_counter = 0;
    double test_accuracy = 0.0;  //evaluate_testing_accuracy();
    double mean_loss = 0.0;


    // Run optimiser - update parameters after each minibatch
    for (int i=0; i < num_batches; i++){
        for (int j = 0; j < batch_size; j++){
            // Evaluate forward pass and calculate gradients
            if (total_iter % log_freq == 0 || total_iter == 0){
                if (total_iter > 0){
                    mean_loss = mean_loss/((double) log_freq);
                }
                test_accuracy = evaluate_testing_accuracy();
                print_training_stats(epoch_counter, total_iter, learning_rate, mean_loss, test_accuracy);

                // save epoch, iter, loss, accuracy to txt file
                FILE *fp;
                fp = fopen("ml.txt", "a");
                fprintf(fp, "%u, %u, %f, %0.12f, %f\n", epoch_counter+1, total_iter, learning_rate, mean_loss, test_accuracy);
                fclose(fp);

                // Reset mean_loss for next reporting period
                mean_loss = 0.0;
            }
            obj_func = evaluate_objective_function(training_sample);
            mean_loss+=obj_func;

            // Update iteration counters (reset at end of training set to allow multiple epochs)
            total_iter++;
            training_sample++;
            // On epoch completion:
            if (training_sample == N_TRAINING_SET){
                training_sample = 0;
                epoch_counter++;

                // Learning Rate Decay
                double alpha = (double)epoch_counter / total_epochs;
                learning_rate = initial_learning_rate * (1 - alpha) + (alpha * final_learning_rate);
            }
        }

        if (mode == 1){
            update_parameters(batch_size);
        }
        else if (mode == 2){
            update_parameters_momentum(batch_size, epoch_counter);
        }
        else if (mode == 3){
            update_parameters_adam(batch_size);
        }
        else {
            printf("Invalid mode selected\n");
        }
    }

    // Print final performance
    test_accuracy = evaluate_testing_accuracy();
    print_training_stats(epoch_counter, total_iter, learning_rate, (mean_loss/((double) log_freq)), test_accuracy);

    // save epoch, iter, loss, accuracy to txt file
    FILE *fp;
    fp = fopen("ml.txt", "a");
    fprintf(fp, "%u, %u, %f, %0.12f, %f\n", epoch_counter+1, total_iter, learning_rate, (mean_loss/((double) log_freq)), test_accuracy);
    fclose(fp);
}

double evaluate_objective_function(unsigned int sample){

    // Compute network performance
    evaluate_forward_pass(training_data, sample);
    double loss = compute_xent_loss(training_labels[sample]);
    
    // Evaluate gradients
    //evaluate_backward_pass(training_labels[sample], sample);
    evaluate_backward_pass_sparse(training_labels[sample], sample);
    
    // Evaluate parameter updates
    evaluate_weight_updates();
    // store_gradient_contributions();
    
    return loss;
}


void update_parameters(unsigned int batch_size){
    // Input->L1
    for (int i=0; i < N_NEURONS_LI; i++){
        for (int j=0; j <N_NEURONS_L1; j++){
            w_LI_L1[i][j].w = w_LI_L1[i][j].w - ((learning_rate/batch_size) * w_LI_L1[i][j].dw);
            w_LI_L1[i][j].dw = 0.0;
        }
    }
    
    // L1->L2
    for (int i=0; i < N_NEURONS_L1; i++){
        for (int j=0; j <N_NEURONS_L2; j++){
            w_L1_L2[i][j].w = w_L1_L2[i][j].w - ((learning_rate/batch_size) * w_L1_L2[i][j].dw);
            w_L1_L2[i][j].dw = 0.0;
        }
    }
    
    // L2->L3
    for (int i=0; i < N_NEURONS_L2; i++){
        for (int j=0; j <N_NEURONS_L3; j++){
            w_L2_L3[i][j].w = w_L2_L3[i][j].w - ((learning_rate/batch_size) * w_L2_L3[i][j].dw);
            w_L2_L3[i][j].dw = 0.0;
        }
    }
    
    // L3-Output
    for (int i=0; i < N_NEURONS_L3; i++){
        for (int j=0; j <N_NEURONS_LO; j++){
            w_L3_LO[i][j].w = w_L3_LO[i][j].w - ((learning_rate/batch_size) * w_L3_LO[i][j].dw);
            w_L3_LO[i][j].dw = 0.0;
        }
    }
}


void update_parameters_momentum(unsigned int batch_size, unsigned int epoch_counter){

    // Input->L1
    for (int i=0; i < N_NEURONS_LI; i++){
        for (int j=0; j <N_NEURONS_L1; j++){

            w_LI_L1[i][j].v = (alpha * w_LI_L1[i][j].v) - (((learning_rate/batch_size) * w_LI_L1[i][j].dw));
            w_LI_L1[i][j].w = w_LI_L1[i][j].w + w_LI_L1[i][j].v;
            w_LI_L1[i][j].dw = 0.0;
        }
    }
    
    // L1->L2
    for (int i=0; i < N_NEURONS_L1; i++){
        for (int j=0; j <N_NEURONS_L2; j++){
            w_L1_L2[i][j].v = (alpha * w_L1_L2[i][j].v) - (((learning_rate/batch_size) * w_L1_L2[i][j].dw));
            w_L1_L2[i][j].w = w_L1_L2[i][j].w + w_L1_L2[i][j].v;
            w_L1_L2[i][j].dw = 0.0;
        }
    }
    
    // L2->L3
    for (int i=0; i < N_NEURONS_L2; i++){
        for (int j=0; j <N_NEURONS_L3; j++){
            w_L2_L3[i][j].v = (alpha * w_L2_L3[i][j].v) - (((learning_rate/batch_size) * w_L2_L3[i][j].dw));
            w_L2_L3[i][j].w = w_L2_L3[i][j].w + w_L2_L3[i][j].v;
            w_L2_L3[i][j].dw = 0.0;
        }
    }
    
    // L3-Output
    for (int i=0; i < N_NEURONS_L3; i++){
        for (int j=0; j <N_NEURONS_LO; j++){
            w_L3_LO[i][j].v = (alpha * w_L3_LO[i][j].v) - (((learning_rate/batch_size) * w_L3_LO[i][j].dw));
            w_L3_LO[i][j].w = w_L3_LO[i][j].w + w_L3_LO[i][j].v;
            w_L3_LO[i][j].dw = 0.0;
        }
    }
}


void update_parameters_adam(unsigned int batch_size){
    double beta_1 = 0.9;
    double beta_2 = 0.9999;
    double epsilon = 0.00000001;

    double bias_corrected_mean;
    double bias_corrected_variance;

    // Input->L1
    for (int i=0; i < N_NEURONS_LI; i++){
        for (int j=0; j <N_NEURONS_L1; j++){
            w_LI_L1[i][j].mean = (beta_1 * w_LI_L1[i][j].mean) + ((1 - beta_1) * w_LI_L1[i][j].dw);
            w_LI_L1[i][j].variance = (beta_2 * w_LI_L1[i][j].variance) + ((1 - beta_2) * (w_LI_L1[i][j].dw * w_LI_L1[i][j].dw));
            
            bias_corrected_mean = (w_LI_L1[i][j].mean / (1 - beta_1));
            bias_corrected_variance = (w_LI_L1[i][j].variance / (1 - beta_2));

            w_LI_L1[i][j].w = w_LI_L1[i][j].w - ((learning_rate/((batch_size * sqrt(bias_corrected_variance)) + epsilon)) * bias_corrected_mean);
            w_LI_L1[i][j].dw = 0.0;
        }
    }
    
    // L1->L2
    for (int i=0; i < N_NEURONS_L1; i++){
        for (int j=0; j <N_NEURONS_L2; j++){
            w_L1_L2[i][j].mean = (beta_1 * w_L1_L2[i][j].mean) + ((1 - beta_1) * w_L1_L2[i][j].dw);
            w_L1_L2[i][j].variance = (beta_2 * w_L1_L2[i][j].variance) + ((1 - beta_2) * (w_L1_L2[i][j].dw * w_L1_L2[i][j].dw));
            
            bias_corrected_mean = (w_L1_L2[i][j].mean / (1 - beta_1));
            bias_corrected_variance = (w_L1_L2[i][j].variance / (1 - beta_2));
            
            w_L1_L2[i][j].w = w_L1_L2[i][j].w - ((learning_rate/((batch_size * sqrt(bias_corrected_variance)) + epsilon)) * bias_corrected_mean);
            w_L1_L2[i][j].dw = 0.0;
        }
    }
    
    // L2->L3
    for (int i=0; i < N_NEURONS_L2; i++){
        for (int j=0; j <N_NEURONS_L3; j++){
            w_L2_L3[i][j].mean = (beta_1 * w_L2_L3[i][j].mean) + ((1 - beta_1) * w_L2_L3[i][j].dw);
            w_L2_L3[i][j].variance = (beta_2 * w_L2_L3[i][j].variance) + ((1 - beta_2) * (w_L2_L3[i][j].dw * w_L2_L3[i][j].dw));
            
            bias_corrected_mean = (w_L2_L3[i][j].mean / (1 - beta_1));
            bias_corrected_variance = (w_L2_L3[i][j].variance / (1 - beta_2));
            
            w_L2_L3[i][j].w = w_L2_L3[i][j].w - ((learning_rate/((batch_size * sqrt(bias_corrected_variance)) + epsilon)) * bias_corrected_mean);
            w_L2_L3[i][j].dw = 0.0;
        }
    }
    
    // L3-Output
    for (int i=0; i < N_NEURONS_L3; i++){
        for (int j=0; j <N_NEURONS_LO; j++){
            w_L3_LO[i][j].mean = (beta_1 * w_L3_LO[i][j].mean) + ((1 - beta_1) * w_L3_LO[i][j].dw);
            w_L3_LO[i][j].variance = (beta_2 * w_L3_LO[i][j].variance) + ((1 - beta_2) * (w_L3_LO[i][j].dw * w_L3_LO[i][j].dw));
            
            bias_corrected_mean = (w_L3_LO[i][j].mean / (1 - beta_1));
            bias_corrected_variance = (w_L3_LO[i][j].variance / (1 - beta_2));
            
            w_L3_LO[i][j].w = w_L3_LO[i][j].w - ((learning_rate/((batch_size * sqrt(bias_corrected_variance)) + epsilon)) * bias_corrected_mean);
            w_L3_LO[i][j].dw = 0.0;
        }
    }
}

void numerical_solution(unsigned int sample){
    w_L1_L2[0][0].dw = 0;
    double old_loss = evaluate_objective_function(sample);
    w_L1_L2[0][0].w = w_L1_L2[0][0].w + 0.15;
    w_L1_L2[0][0].dw = 0;
    double new_loss = evaluate_objective_function(sample+1);
    double numeric = (learning_rate/batch_size) * ((new_loss - old_loss)/0.15);
    double analytic = w_L1_L2[0][0].dw;
    double analytic2 = dL_dW_L1_L2[0][0];
    printf("Numerical Solution = %f, Analytical Solution = %f, Analytical Solution 2 = %f\n", numeric, analytic, analytic2);
}