
#include <stdio.h>
#include <string.h>
#include <math.h>


#include "rlglue/Agent_common.h" /* Required for RL-Glue */
#include "utils.h"
#include "common.h"

//#define EPSILON		0.1 /* Exploration probability */
//#define	ALPHA		0.5 /* Learning rate */
//#define GAMMA		1.0 /* Discounting factor */
//#define Q_TERMINAL	0.0 /* Q-Value for terminal state */


static gsl_vector* local_action;
static action_t* this_action;
static gsl_vector* last_observation;

static int **last_n_observations;

static double Q[GRIDWIDTH][GRIDHEIGHT][N_ACTIONS];
static double *R;

static double G;

static int t;
static int tou;

static const double epsilon = 0.1;
static const double alpha = 0.8;
static const double disc_factor = 1.0;
static const double q_terminal = 0.0;

static const int n = 5;

void agent_init()
{

	//Allocate Memory

	local_action = gsl_vector_calloc(1);
	this_action = local_action;
	last_observation = gsl_vector_calloc(2);
	
	R = (double *)calloc(n, sizeof(double));

	last_n_observations = (int **)calloc(n, sizeof(int *));

	for(int i = 0; i < n; i++){
		last_n_observations[i] = (int *)calloc(3, sizeof(int));
		last_n_observations[i][0] = START_X;
		last_n_observations[i][1] = START_Y;
		last_n_observations[i][2] = 0;
	}

	for(int i = 0; i < GRIDWIDTH; i++){
		for(int j = 0; j < GRIDHEIGHT; j++){
			for(int k = 0; k < N_ACTIONS; k++)
				Q[i][j][k] = 0;

		}
	}
	
}

const action_t *agent_start(const observation_t *this_observation) {
 	
  	//Read State

	int x = (int)gsl_vector_get(this_observation, 0);
	int y = (int)gsl_vector_get(this_observation, 1);
	
	//Get optimal and suboptimal actions
	int opt_actions[N_ACTIONS] = {0, 0, 0, 0, 0, 0, 0, 0};
	int sub_opt_actions[N_ACTIONS] = {0, 0, 0, 0, 0, 0, 0, 0};

	int maxcount = 0;
	int subcount = 0;
	
	double maxval = Q[x][y][0];
	for(int i = 1; i < N_ACTIONS; i++){
		if(maxval < Q[x][y][i])
			maxval = Q[x][y][i];
	}

	for(int i = 0; i < N_ACTIONS; i++){
		if(Q[x][y][i] == maxval)
			opt_actions[maxcount++] = i;
		else
			sub_opt_actions[subcount++] = i;
	}
	

	//Randomly select optimal action
	int maxindex = randInRange(maxcount);
	int opt_act = opt_actions[(maxindex < N_ACTIONS) ? maxindex : N_ACTIONS - 1];

	//Randomly select suboptimal action
	int subindex = randInRange(subcount);
	int sub_opt_act = sub_opt_actions[(subindex < N_ACTIONS) ? subindex : N_ACTIONS - 1];
	
	//Select action based on epsilon-greedy policy
	double p = rand_un();

	int act = randInRange(N_ACTIONS);

	act = (act < N_ACTIONS) ? act : N_ACTIONS - 1;
	
	if(p >= epsilon || subcount == 0)
		act = opt_act;
	else
		act = sub_opt_act;
	
	//Save action in local_action
	gsl_vector_set(local_action, 0, act);

	//Save last observation
	gsl_vector_memcpy(last_observation, this_observation);
	
	tou = 0;
	t = 0;
	G = 0.0;

	last_n_observations[n-1][0] = x;
	last_n_observations[n-1][1] = y;
	last_n_observations[n-1][2] = act;


  	return this_action;
}

const action_t *agent_step(double reward, const observation_t *this_observation) {
	
	/* Code to update R - START */
	for(int i = 0; i < n-1; i++)
		R[i] = R[i + 1];
	R[n-1] = reward;
	/* Code to update R - END */

	//Read State
	int x_prime = (int)gsl_vector_get(this_observation, 0);
	int y_prime = (int)gsl_vector_get(this_observation, 1);
	
		
	//Get set of optimal and suboptimal actions

	int opt_actions[N_ACTIONS] = {0, 0, 0, 0, 0, 0, 0, 0};
	int sub_opt_actions[N_ACTIONS] = {0, 0, 0, 0, 0, 0, 0, 0};


	int maxcount = 0;
	int subcount = 0;

	double maxval = Q[x_prime][y_prime][0];
	
	for(int i = 1; i < N_ACTIONS; i++){
		if(maxval < Q[x_prime][y_prime][i])
			maxval = Q[x_prime][y_prime][i];

	}

	for(int i = 0; i < N_ACTIONS; i++){
		if(Q[x_prime][y_prime][i] >= maxval)
			opt_actions[maxcount++] = i;
		else
			sub_opt_actions[subcount++] = i;
	}
	
	//Randomly select optimal action
	int maxindex = randInRange(maxcount);
	int opt_act = opt_actions[(maxindex < N_ACTIONS) ? maxindex : N_ACTIONS - 1];

	//Randomly select suboptimal action
	int subindex = randInRange(subcount);
	int sub_opt_act = sub_opt_actions[(subindex < N_ACTIONS) ? subindex : N_ACTIONS - 1];
	

	//Select action based on epsilon-greedy policy
	double p = rand_un();

	int act_prime = randInRange(N_ACTIONS);

	act_prime = (act_prime < N_ACTIONS) ? act_prime : N_ACTIONS - 1;
	
	if(p >= epsilon || subcount == 0)
		act_prime = opt_act;
	else
		act_prime = sub_opt_act;
	
	
	tou = t - n + 1;
	
	if(tou >= 0){
		/* Code to update G - START*/
		
		G += R[n-1];

		for(int i = n-2; i > 0; i--){
			G = R[i] + disc_factor * G;
		}

		G += R[0];

		/* Code to update G - END */
		
		G += pow(disc_factor, (double)n) * Q[x_prime][y_prime][act_prime];

		/* Code to update Q(S_tou, A_tou) - START*/
		int x_tou = last_n_observations[0][0];
		int y_tou = last_n_observations[0][1];
		int act_tou = last_n_observations[0][2];

		Q[x_tou][y_tou][act_tou] += alpha * (G - Q[x_tou][y_tou][act_tou]);
		/* Code to update Q(S_tou, A_tou) - END */

	}
	
	//Save action in local_action
	gsl_vector_set(local_action, 0, act_prime);

	//Save last observation
	gsl_vector_memcpy(last_observation, this_observation);
	
	t++;
	
	G = 0.0;

	/*  Update last n observations - START*/
	for(int i = 0; i < n-1; i++){
		last_n_observations[i][0] = last_n_observations[i + 1][0];
		last_n_observations[i][1] = last_n_observations[i + 1][1];
		last_n_observations[i][2] = last_n_observations[i + 1][2];
	}
	last_n_observations[n-1][0] = x_prime;
	last_n_observations[n-1][1] = y_prime;
	last_n_observations[n-1][2] = act_prime;
	/*  Update last n observations - END*/


  	return this_action;
}


void agent_end(double reward) {
  /* final learning update at end of episode */

	//Get last action
	int act = (int)gsl_vector_get(local_action, 0);
  	
	//Get last observation
	int x = (int)gsl_vector_get(last_observation, 0);
	int y = (int)gsl_vector_get(last_observation, 1);

	tou = t - n + 1;

	if(tou >= 0){
		/* Code to update G - START*/
		
		G += R[n-1];

		for(int i = n-2; i > 0; i--){
			G = R[i] + disc_factor * G;
		}

		G += R[0];

		/* Code to update G - END */

		/* Code to update Q(S_tou, A_tou) - START*/
		int x_tou = last_n_observations[0][0];
		int y_tou = last_n_observations[0][1];
		int act_tou = last_n_observations[0][2];

		Q[x_tou][y_tou][act_tou] += alpha * (G - Q[x_tou][y_tou][act_tou]);
		/* Code to update Q(S_tou, A_tou) - END */

		
	}

}

void agent_cleanup() {
  /* clean up mememory */
  gsl_vector_free(local_action);
  gsl_vector_free(last_observation);
}

const char* agent_message(const char* inMessage) {
  /* might be useful to get information from the agent */
  if(strcmp(inMessage,"HELLO")==0)
  return "THIS IS SARSA";
  
  /* else */
  return "I don't know how to respond to your message";
}



/*
#ifdef EXPECTED_SARSA
static gsl_vector* local_action;
static action_t* this_action;
static gsl_vector* last_observation;

void agent_init()
{
	printf("Expected SARSA Agent\n");
  //NOT USED
}

const action_t *agent_start(const observation_t *this_observation) {
 	
  //NOT USED
  
  return this_action;
}

const action_t *agent_step(double reward, const observation_t *this_observation) {
  

  	//NOT USED
  
  	return this_action;
}


void agent_end(double reward) {
  // final learning update at end of episode
}

void agent_cleanup() {
  //clean up mememory
  gsl_vector_free(local_action);
  gsl_vector_free(last_observation);
}

const char* agent_message(const char* inMessage) {
  //might be useful to get information from the agent
  if(strcmp(inMessage,"HELLO")==0)
  return "THIS IS SARSA";
  
  // else
  return "I don't know how to respond to your message";
}
#endif

*/
