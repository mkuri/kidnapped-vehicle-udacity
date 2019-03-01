/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;

static std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; ++i) {
    Particle p;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;

    particles.push_back(p);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  for (int i = 0; i < num_particles; ++i) {
    double dx, dy, dtheta, theta0;
    theta0 = particles[i].theta;
    dtheta = yaw_rate * delta_t;
    if (fabs(yaw_rate) < 0.0001) {
      dx = velocity * delta_t * cos(theta0);
      dy = velocity * delta_t * sin(theta0);
    } else {
      dx = velocity/yaw_rate * (sin(theta0+dtheta) - sin(theta0));
      dy = velocity/yaw_rate * (cos(theta0) - cos(theta0+dtheta));
    }
    normal_distribution<double> dist_x(particles[i].x + dx, std_pos[0]);
    normal_distribution<double> dist_y(particles[i].y + dy, std_pos[1]);
    normal_distribution<double> dist_theta(particles[i].theta + dtheta, std_pos[2]);
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

  for (unsigned int i = 0; i < observations.size(); ++i) {
    LandmarkObs obs = observations[i];

    double min_distance = std::numeric_limits<double>::max();
    LandmarkObs pred;
    double distance;
    for (int j = 0; j < predicted.size(); ++j) {
      pred = predicted[j];
      distance = dist(obs.x, obs.y, pred.x, pred.y);
      if (distance < min_distance) {
        min_distance = distance;
        observations[i].id = pred.id;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  double p_x, p_y, p_theta;
  for (int i = 0; i < num_particles; ++i) {
    p_x = particles[i].x;
    p_y = particles[i].y;
    p_theta = particles[i].theta;
    particles[i].weight = 1.0;

    // Convert observations to the map coordinates
    vector<LandmarkObs> map_observations;
    double obs_x, obs_y, obs_mapx, obs_mapy;
    for (unsigned int j = 0; j < observations.size(); ++j) {
      obs_x = observations[j].x;
      obs_y = observations[j].y;
      obs_mapx = cos(p_theta)*obs_x - sin(p_theta)*obs_y + p_x;
      obs_mapy = sin(p_theta)*obs_x + cos(p_theta)*obs_y + p_y;
      map_observations.push_back(LandmarkObs{observations[j].id, obs_mapx, obs_mapy});
    }

    // Select landmarks in sensor range
    vector<LandmarkObs> landmarks_in_range;
    float lm_x, lm_y;
    int lm_id;
    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
      lm_x = map_landmarks.landmark_list[j].x_f;
      lm_y = map_landmarks.landmark_list[j].y_f;
      lm_id = map_landmarks.landmark_list[j].id_i;
      if (dist(lm_x, lm_y, p_x, p_y) <= sensor_range) {
        landmarks_in_range.push_back(LandmarkObs{lm_id, lm_x, lm_y});
      }
    }

    // Associate observations with landmarks
    dataAssociation(landmarks_in_range, map_observations);

    int map_id;
    double std_x, std_y, gauss_norm, exponent, obs_weight;
    for (int j = 0; j < map_observations.size(); ++j) {
      obs_mapx = map_observations[j].x;
      obs_mapy = map_observations[j].y;
      map_id = map_observations[j].id;

      for (int k = 0; k < landmarks_in_range.size(); ++k) {
        if (landmarks_in_range[k].id == map_id) {
          lm_x = landmarks_in_range[k].x;
          lm_y = landmarks_in_range[k].y;
        }
      }

      std_x = std_landmark[0];
      std_y = std_landmark[1];
      gauss_norm = 1 / (2 * M_PI * std_x * std_y);
      exponent = (pow(obs_mapx - lm_x, 2) / (2 * pow(std_x, 2))) + (pow(obs_mapy - lm_y, 2) / (2 * pow(std_y, 2)));
      obs_weight = gauss_norm * exp(-exponent);

      particles[i].weight *= obs_weight;
    }
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  vector<Particle> resampled_particles(num_particles);
  vector<double> weights;
  for (int i = 0; i < num_particles; ++i) {
    weights.push_back(particles[i].weight);
  }
  std::discrete_distribution<int> dist(weights.begin(), weights.end());

  int index;
  for (int i = 0; i < num_particles; ++i) {
    index = dist(gen);
    resampled_particles[i] = particles[index];
  }

  particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
