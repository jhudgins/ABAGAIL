package opt;

import shared.Instance;

/**
 * Keep track of stats for optimization fitness
 * @author Jonathan Hudgins jhudgins8@gatech.edu
 */
public class FitnessStats
{
    public double localBestFitness;
    public double fitness;
    public int changeDimension;
    public double delta;
    public double newValue;
    public FitnessStats(double localBestFitness, double fitness, int changeDimension, double delta, double newValue)
    {
        this.localBestFitness = localBestFitness;
        this.fitness = fitness;
        this.changeDimension = changeDimension;
        this.delta = delta;
        this.newValue = newValue;
    }
}
 
