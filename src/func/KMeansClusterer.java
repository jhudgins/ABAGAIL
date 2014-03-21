package func;

import java.text.DecimalFormat;

import shared.DataSet;
import shared.DistanceMeasure;
import shared.EuclideanDistance;
import shared.Instance;
import util.linalg.DenseVector;
import util.linalg.Vector;
import dist.AbstractConditionalDistribution;
import dist.DiscreteDistribution;
import dist.Distribution;

/**
 * A K means clusterer
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class KMeansClusterer extends AbstractConditionalDistribution implements FunctionApproximater {
    /**
     * The cluster centers
     */
    private Instance[] clusterCenters;
    private int[] assignments;
    private double[] assignmentCount;

    // useful stats 
    double[] meanDist;
    double[] minDist;
    double[] maxDist;
    double[] varience;

    double[] meanNextClosest;
    double[] minNextClosest;
    double[] maxNextClosest;

    double[] volume;
 
    int iterations;

    private static DecimalFormat df = new DecimalFormat("0.00000");
    
    /**
     * The number of clusters
     */
    private int k;
    
    /**
     * The distance measure
     */
    private DistanceMeasure distanceMeasure;
    
    /**
     * Make a new k means clustere
     * @param k the k value
     * @param distanceMeasure the distance measure
     */
    public KMeansClusterer(int k, DistanceMeasure distanceMeasure) {
        this.k = k;
        this.distanceMeasure = distanceMeasure;
    }

    public KMeansClusterer(int k) {
    	this(k, new EuclideanDistance());
    }
    
    /**
     * Make a new clusterer
     */
    public KMeansClusterer() {
        this(2);
    }

    /**
     * @see func.Classifier#classDistribution(shared.Instance)
     */
    public Distribution distributionFor(Instance instance) {
        double[] distribution = new double[k];
        for (int i = 0; i < k; i++) {
            distribution[i] +=
                1/distanceMeasure.value(instance, clusterCenters[i]);   
        }
        double sum = 0;
        for (int i = 0; i < distribution.length; i++) {
            sum += distribution[i];
        }
        if (Double.isInfinite(sum)) {
            sum = 0;
            for (int i = 0; i < distribution.length; i++) {
                if (Double.isInfinite(distribution[i])) {
                    distribution[i] = 1;
                    sum ++;
                } else {
                    distribution[i] = 0;
                }
            }
        }
        for (int i = 0; i < distribution.length; i++) {
            distribution[i] /= sum;
        }
        return new DiscreteDistribution(distribution);
    }

    /**
     * @see func.FunctionApproximater#estimate(shared.DataSet)
     */
    public void estimate(DataSet set) {
        estimate(set, 1000, 0.01);
    }

    public void estimate(DataSet set, int maxIterations, double stopAtProportionChanged) {
        clusterCenters = new Instance[k];
        assignmentCount = new double[k];
        assignments = new int[set.size()];
        // random initial centers
        for (int i = 0; i < clusterCenters.length; i++) {
            int pick;
            do {
                pick = Distribution.random.nextInt(set.size());
            } while (assignments[pick] != 0);
            assignments[pick] = i;
            clusterCenters[i] = (Instance) set.get(pick).copy();
        }
        int changed = set.size();
        iterations = 0;
        // the main loop
        while (changed > stopAtProportionChanged * set.size() && iterations < maxIterations) {
            changed = 0;
            for (int cluster=0; cluster<k; cluster++) {
            	assignmentCount[cluster] = 0;
            }
            // make the assignments
            for (int i = 0; i < set.size(); i++) {
                // find the closest center
                int closest = 0;
                double closestDistance = distanceMeasure
                    .value(set.get(i), clusterCenters[0]);
                for (int j = 1; j < k; j++) {
                    double distance = distanceMeasure
                        .value(set.get(i), clusterCenters[j]);
                    if (distance < closestDistance) {
                        closestDistance = distance;
                        closest = j;
                    }
                }
                if (assignments[i] != closest) {
                    changed++;
                }
                assignments[i] = closest;
            }
            if (changed > 0) {
                // make the new clusters
                for (int i = 0; i < k; i++) {
                    clusterCenters[i].setData(new DenseVector(
                        clusterCenters[i].getData().size()));
                }
                for (int i = 0; i < set.size(); i++) {
                    clusterCenters[assignments[i]].getData().plusEquals(
                        set.get(i).getData().times(set.get(i).getWeight()));
                    assignmentCount[assignments[i]] += set.get(i).getWeight();    
                }
                for (int i = 0; i < k; i++) {
                    clusterCenters[i].getData().timesEquals(1/assignmentCount[i]);
                }
            }
            iterations++;
        }

        // do some descriptive analysis
        meanDist = new double[k];
        minDist = new double[k];
        maxDist = new double[k];
        varience = new double[k];

        meanNextClosest = new double[k];
        minNextClosest = new double[k];
        maxNextClosest = new double[k];

        volume = new double[k];
        Vector minDim[] = new Vector[k];
        Vector maxDim[] = new Vector[k];
        int dimensions = set.get(0).getData().size();

        // intialize min max values
        for (int cluster=0; cluster<k; cluster++) {
            minDist[cluster] = Double.MAX_VALUE;
            minNextClosest[cluster] = Double.MAX_VALUE;
            maxDist[cluster] = -Double.MAX_VALUE;
            maxNextClosest[cluster] = -Double.MAX_VALUE;

            double[] minDimValues = new double[dimensions];
            double[] maxDimValues = new double[dimensions];
            for (int j=0; j<dimensions; j++) {
                minDimValues[j] = Double.MAX_VALUE;
                maxDimValues[j] = -Double.MAX_VALUE;
            }
            minDim[cluster] = new DenseVector(minDimValues);
            maxDim[cluster] = new DenseVector(maxDimValues);
        }

        // calculate first pass values
        for (int i=0; i<set.size(); i++) {
            int cluster = assignments[i];

            // calculate distance to centroid
            double distance = distanceMeasure.value(set.get(i), clusterCenters[cluster]);
            meanDist[cluster] += distance;
            minDist[cluster] = Math.min(minDist[cluster], distance);
            maxDist[cluster] = Math.max(maxDist[cluster], distance);

            // collect min and max of each dimension for this cluster
            for (int dim=0; dim<dimensions; dim++) {
                minDim[cluster].set(dim, Math.min(minDim[cluster].get(dim), set.get(i).getContinuous(dim)));
                maxDim[cluster].set(dim, Math.max(maxDim[cluster].get(dim), set.get(i).getContinuous(dim)));
            }

            // find closest centroid (not my own)
            double closestDist = Double.MAX_VALUE;
            for (int j=0; j<k; j++) {
                if (j != cluster) {
                    closestDist = Math.min(closestDist, distanceMeasure.value(set.get(i), clusterCenters[j]));
                }
            }

            meanNextClosest[cluster] += closestDist;
            minNextClosest[cluster] = Math.min(closestDist, minNextClosest[cluster]);
            maxNextClosest[cluster] = Math.max(closestDist, maxNextClosest[cluster]);
        }

        // divide to get means and calculate volume
        double maxVolume = 0.;
        for (int cluster=0; cluster<k; cluster++) {
            meanDist[cluster] /= assignmentCount[cluster];
            meanNextClosest[cluster] /= assignmentCount[cluster];
            Vector delta = maxDim[cluster].minus(minDim[cluster]);
            volume[cluster] = 1.;
            for (int dim=0; dim<dimensions; dim++) {
                volume[cluster] *= delta.get(dim);
            }
            maxVolume = Math.max(maxVolume, volume[cluster]);
        }

        // normalize volumes
        for (int cluster=0; cluster<k; cluster++) {
            volume[cluster] /= maxVolume;
        }

        // calculate varience
        for (int i=0; i<set.size(); i++) {
            int cluster = assignments[i];
            assert cluster < k && cluster > 0;

            // calculate distance to centroid
            double distance = distanceMeasure.value(set.get(i), clusterCenters[cluster]);
            double delta = distance - meanDist[cluster];
            varience[cluster] += (delta * delta) / assignmentCount[cluster];
        }
    }

    /**
     * @see func.FunctionApproximater#value(shared.Instance)
     */
    public Instance value(Instance data) {
        return distributionFor(data).mode();
    }

    /**
     * Get the cluster centers
     * @return the cluster centers
     */
    public Instance[] getClusterCenters() {
        return clusterCenters;
    }
        
    /**
     * @see java.lang.Object#toString()
     */
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("k = " + k + ", iterations = " + iterations + "\n");
        stringBuilder.append("instances,meanToCentroid,minToCentroid,maxToCentroid,stdDevToCentroid,"
                             + "meanToNextCentroid,minToNextCentroid,maxToNextCentroid,normalizedVolume\n");

        for (int cluster=0; cluster<k; cluster++) {
            stringBuilder.append(assignmentCount[cluster] + ",");
            stringBuilder.append(df.format(meanDist[cluster]) + ",");
            stringBuilder.append(df.format(minDist[cluster]) + ",");
            stringBuilder.append(df.format(maxDist[cluster]) + ",");
            stringBuilder.append(df.format(Math.sqrt(varience[cluster])) + ",");

            stringBuilder.append(df.format(meanNextClosest[cluster]) + ",");
            stringBuilder.append(df.format(minNextClosest[cluster]) + ",");
            stringBuilder.append(df.format(maxNextClosest[cluster]) + ",");

            stringBuilder.append(df.format(volume[cluster]) + "\n");
        }

        return stringBuilder.toString();
    }
}
