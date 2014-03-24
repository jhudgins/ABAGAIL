package shared;



/**
 * Standard error measure, suitable for use with
 * linear output networks for regression, sigmoid
 * output networks for single class probability,
 * and soft max networks for multi class probabilities.
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class SumOfSquaresError extends AbstractErrorMeasure
        implements GradientErrorMeasure {

    /**
     * @see nn.error.ErrorMeasure#error(double[], nn.Pattern[], int)
     */
    public double value(Instance output, Instance example) {
        double sum = 0;
        Instance label = example.getLabel();
        int labelIndex = label.getDiscrete();
        for (int i = 0; i < output.size(); i++) {
            // output should be 1 on labelIndex, 0 otherwise for "best fit"
            double delta;
            if (i == labelIndex) {
                delta = 1.0 - output.getContinuous(i);
            }
            else {
                delta = 0.0 - output.getContinuous(i);
            }
            
            sum += delta * delta * example.getWeight();
        }
        // divide by 2 to match gradient descent ANN derivation
        return .5 * sum;
    }

    /**
     * @see nn.error.DifferentiableErrorMeasure#derivatives(double[], nn.Pattern[], int)
     */
    public double[] gradient(Instance output, Instance example) {      
        double[] errorArray = new double[output.size()];
        Instance label = example.getLabel();
        int labelIndex = label.getDiscrete();
        for (int i = 0; i < output.size(); i++) {
            double delta;
            if (i == labelIndex) {
                delta = output.getContinuous(i) - 1.0;
            } else {
                delta = output.getContinuous(i) - 0.0;
            }
            errorArray[i] = delta * example.getWeight();
        }
        return errorArray;
    }

}
