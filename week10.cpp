#include <iostream>

// linear regression < deep learning < machine learning

class LinearHypothesis
{
public:
	// linear hypothesis : y = a * x + b
	float a_ = 0.0f;
	float b_ = 0.0f;

	float getY(const float& x_input)
	{
		const float y = a_ * x_input + b_; // returns y = a*x+b

		return y;
	}
};

const int num_data = 100000;

int main()
{
	// 0 hour -> 0 pts
	// 1 hour -> 2 pts
	// 2 hour -> 4 pts
	// 2.5 hour -> ? (guess)

	float study_time_data[num_data];
	float score_data[num_data];

	for (int i = 0; i < num_data; i++)
	{
		float tmp = rand() / (float)RAND_MAX * 10;
		//float tmp = rand() %20;
		study_time_data[i] = tmp;
		score_data[i] = 5 * tmp + 2;
	}


	// input x is study time -> black box(AI) -> output y is score
	// linear hypothesis : y = a * x + b
	LinearHypothesis lh;

	for (int tr = 0; tr < 100; tr++)
		for (int i = 0; i < num_data; i++)
		{
			// let's train our linear hypothesis to answer correctly
			const float x_input = study_time_data[i];
			const float y_output = lh.getY(x_input);
			const float y_target = score_data[i];
			const float error = y_output - y_target;
			// we can consider that our LH is trained well when error is 0 or small enough
			// we define squared error 
			const float sqr_error = 0.5 * error * error;

			// we want to find good combination of a and b which minimizes sqr_error

			// sqr_error = 0.5 * (a * x + b - y_target)^2
			// d sqr_error / da = 2*0/5*(a * x +b - y_target) * x;
			// d sqr_error / db = 2*0/5*(a * x +b - y_target) * 1;

			const float dse_over_da = error * x_input;
			const float dse_over_db = error;

			// need to find good a and b
			// we can update a and b
			// this is the gradient descent method

			const float lr = 0.01; // small number
			lh.a_ -= dse_over_da * lr;
			lh.b_ -= dse_over_db * lr;

			//std::cout << "x_input = " << x_input << "y_target = " << y_target << "y_output = " << y_output << "sqr_error = " << sqr_error << std::endl;
		}

	//trained hypothesis
	std::cout << "y = 5*x + 2" << std::endl;
	std::cout << "a = " << lh.a_ << std::endl;
	std::cout << "b = " << lh.b_ << std::endl;

	//std::cout << "From trained hypothesis " << lh.getY(2.5) << std::endl;

	return 0;
}
