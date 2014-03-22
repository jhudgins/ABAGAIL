package shared.filt;

import util.linalg.Matrix;

/**
 * A Projection Filter
 * @author Jonathan Hudgins jhudgins8@gatech.edu
 * @version 1.0
 */
public interface ProjectionFilter extends ReversibleFilter {
    /**
     * Get the projection matrix used to transform the attributes
     * @return projection matrix
     */
    public Matrix getProjection();

}
