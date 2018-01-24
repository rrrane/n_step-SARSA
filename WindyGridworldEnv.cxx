
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "rlglue/Environment_common.h" /* Required for RL-Glue interface*/

#include "common.h"

static gsl_vector* local_observation;
static reward_observation_terminal_t this_reward_observation;

static int wind_vector[] = {0, 0, 0, 1, 1, 1, 2, 2, 1, 0};

void env_init() 
{
	local_observation = gsl_vector_calloc(2);
	this_reward_observation.observation = local_observation;
	this_reward_observation.reward = 0;
	this_reward_observation.terminal = 0;

	return;
}

const observation_t* env_start()
{
	gsl_vector_set(local_observation, 0, START_X);
	gsl_vector_set(local_observation, 1, START_Y);
	
  	return this_reward_observation.observation;
}

const reward_observation_terminal_t *env_step(const action_t *this_action)
{
	
	double the_reward = -1.0;

	int episode_over = 0;

	int x = (int)gsl_vector_get(local_observation, 0);
	int y = (int)gsl_vector_get(local_observation, 1);

	int act = (int)gsl_vector_get(this_action, 0);
	//printf("%d ", act);

	if(act > 7)
		printf("FROM ENV: ACTION EXCEEDED LIMIT (%d)\n", act);

	int dx = (act == LEFT || act == UP_LEFT || act == DOWN_LEFT) ? -1 : (act == RIGHT || act == UP_RIGHT || act == DOWN_RIGHT) ? 1 : 0;
	int dy = (act == UP || act == UP_RIGHT || act == UP_LEFT) ? 1 : (act == DOWN || act == DOWN_RIGHT || act == DOWN_LEFT) ? -1 : 0;

	int new_x = x + dx;
	int new_y = y + dy + wind_vector[x];

	new_x = (new_x >= GRIDWIDTH) ? GRIDWIDTH - 1 : (new_x < 0) ? 0 : new_x;
	new_y = (new_y >= GRIDHEIGHT) ? GRIDHEIGHT - 1 : (new_y < 0) ? 0 : new_y;
	
	//printf("%d, %d\n", new_x, new_y);

	gsl_vector_set(local_observation, 0, new_x);
	gsl_vector_set(local_observation, 1, new_y);

	if(new_x == GOAL_X && new_y == GOAL_Y){
		episode_over = 1;
		the_reward = 0.0;
	}

	this_reward_observation.reward = the_reward;
	this_reward_observation.terminal = episode_over;

  
  	return &this_reward_observation;
}


void env_cleanup()
{
  gsl_vector_free(local_observation);
}

const char* env_message(const char* inMessage) 
{
  if(strcmp(inMessage,"what is your name?")==0)
  return "my name is skeleton_environment!";
  
  /* else */
  return "I don't know how to respond to your message";
}
