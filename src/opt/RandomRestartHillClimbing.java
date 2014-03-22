package opt;

import java.util.ArrayList;
import java.util.Random;

import shared.Instance;
import util.linalg.Vector;
import util.RandomFactory;

/**
 * Random restart hill climbing
 * @author Jonathan Hudgins jhudgins8@gatech.edu
 * @version 1.0
 *
 * Options:
 *      Restart:
 *          constant
 *          random
 *          converged
 *
 *      Increment: -- understand evaluation
 *          constant
 *          random
 *          converges (try 1/2 increment)
 *
 *      Changing dimension:
 *          random
 *          round-robin
 *          converges
 *
 * Measure preformance:
 *      graph of localBest
 *      restarts
 *      coverage
 *          number of changes by dimension
 *          mean changes per dimension
 *          std. dev changes per dimension
 *          range by dimension
 */


public class RandomRestartHillClimbing extends OptimizationAlgorithm {

    // keep track of local best and global best
    private Instance mLocalBest = null;
    private Instance mGlobalBest = null;
    private double mLocalBestFitness;
    private double mGlobalBestFitness;

    private int mConverganceCount = 0;
    private int mRestarts = 0;

    private int mDimension;

    private OptPolicy mDimensionPolicy;
    private double mImprovementThreshhold;
    private OptPolicy mIncrementPolicy;
    private double mIncrementDefault;
    private OptPolicy mRestartPolicy;
    private double mRestartParam;

    private double[] mIncrementForDim;
    private ArrayList<FitnessStats> mFitnessStats = new ArrayList<FitnessStats>();
    
    private Random mRandom = RandomFactory.newRandom();
    /**
     * Make a new randomized hill climbing
     */
    public RandomRestartHillClimbing(HillClimbingProblem hcp,
                                OptPolicy dimensionPolicy, double improvementThreshhold,
                                OptPolicy incrementPolicy, double incrementDefault,
                                OptPolicy restartPolicy, double restartParam)
    {
        super(hcp);
        // start by setting global and local best to our first random value
        mGlobalBest = mLocalBest = hcp.random();
        mGlobalBestFitness = mLocalBestFitness = hcp.value(mLocalBest);
        mFitnessStats.add(new FitnessStats(mLocalBestFitness, mLocalBestFitness, 0, 0., 0.));

        mDimensionPolicy = dimensionPolicy;
        mImprovementThreshhold = improvementThreshhold;
        mIncrementPolicy = incrementPolicy;
        mIncrementDefault = incrementDefault;
        mRestartPolicy = restartPolicy;
        mRestartParam = restartParam;

        mIncrementForDim = new double[mLocalBest.size()];
        for (int i=0; i<mIncrementForDim.length; i++) {
            mIncrementForDim[i] = mIncrementDefault;
        }
    } 

    private void restart() {
        HillClimbingProblem hcp = (HillClimbingProblem) getOptimizationProblem();
        mLocalBest = hcp.random();
        mLocalBestFitness = hcp.value(mLocalBest);
        mFitnessStats.add(new FitnessStats(mLocalBestFitness, mLocalBestFitness, 0, 0., 0.));
        for (int i=0; i<mIncrementForDim.length; i++) {
            mIncrementForDim[i] = mIncrementDefault;
        }
        mRestarts++;
    }
    /**
     * @see shared.Trainer#train()
     */
    public double train() {
        HillClimbingProblem hcp = (HillClimbingProblem) getOptimizationProblem();

        switch (mDimensionPolicy) {
            case Random:     mDimension = mRandom.nextInt(mLocalBest.size()); break;
            case RoundRobin: mDimension = (mDimension + 1) % mLocalBest.size(); break;

            // when converging each dimension independantly, dimension updated after convergence test below
            case Converge:   break;
        }

        double increment = mIncrementDefault;
        switch (mIncrementPolicy) {
            case Constant:  increment = mIncrementDefault; break;
            case Random:    increment = mIncrementDefault * mRandom.nextDouble(); break;
            case Converge:  increment = mIncrementForDim[mDimension];
        }

        // keep old value to restore after testing neighbors (prevents having to copy)
        Vector data = mLocalBest.getData();
        double keepValue = data.get(mDimension);
        double initialValue = keepValue;
        boolean update = false;

        // test fitness to left and right
        for (int i=0; i<2; i++) {
            // calculate -increment or +increment
            double delta = (2*i-1) * increment;
            double testValue = initialValue + delta;
            data.set(mDimension, testValue);
            double fitness = hcp.value(mLocalBest);

            // if we have fitness better by mImprovementThreshhold proportion, then update keepValue
            // if localfitness is negative, then we want to skew towards 0 (divide by mImprovementThreshhold),
            // otherwise we want to skew away from 0 (multiply by mImprovementThreshhold)
            double improvementThreshhold = mLocalBestFitness < 0 ? 1./mImprovementThreshhold : mImprovementThreshhold;
            if (fitness > mLocalBestFitness * improvementThreshhold) {
                update = true;
                mLocalBestFitness = fitness;
                keepValue = testValue;

                // reset converge to retry
                mConverganceCount = 0;

                // check global optimum
                if (mLocalBestFitness > mGlobalBestFitness) {
                    mGlobalBestFitness = mLocalBestFitness;
                    mGlobalBest = mLocalBest;
                }
            }
            mFitnessStats.add(new FitnessStats(mLocalBestFitness, fitness, mDimension, delta, keepValue));
        }

        // restore the best keepValue
        data.set(mDimension, keepValue);

        if (!update) {
            // we change the increment so that the next two increments would be very close to a full increment
            if (mIncrementPolicy == OptPolicy.Converge) {
                mIncrementForDim[mDimension] *= 0.49;
                if (mIncrementForDim[mDimension] < 0.05 * mIncrementDefault) {
                    mConverganceCount++;
                    mDimension = (mDimension + 1) % mLocalBest.size();
                }
            }
            else if (mDimensionPolicy == OptPolicy.Converge) {
                mDimension = (mDimension + 1) % mLocalBest.size();
                mConverganceCount++;
            }
            else {
                mConverganceCount++;
            }
        }

        switch (mRestartPolicy) {
            case Constant: // restart every (mRestartParam) evaluations
                           if (mFitnessStats.size() > (mRestarts+1) * mRestartParam) { restart(); } break;
            case Random:   // restart every random proportion (mRestartParam)
                           if (mRandom.nextDouble() < mRestartParam) { restart(); } break;
            case Converge: // restart if we have converged in every dimesion
                           if (mConverganceCount >= mLocalBest.size()) { restart(); } break;
        }
                
        return mGlobalBestFitness;
    }

    /**
     * @see opt.OptimizationAlgorithm#getOptimalData()
     */
    public Instance getOptimal() {
        return mGlobalBest;
    }

    /**
     * @return number of time mRestarts were performed
     */
    public int getRestarts() {
        return mRestarts;
    }

    /**
     * @return number of time mEvaluations were performed
     */
    public int getEvaluations() {
        return mFitnessStats.size();
    }

    /**
     * @return ArrayList of fitness at each evaluation
     */
    public ArrayList<FitnessStats> getFitnessStats() {
        return mFitnessStats;
    }

}
