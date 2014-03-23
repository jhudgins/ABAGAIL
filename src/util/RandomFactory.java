package util;

import java.util.Random;

/**
 * Creates a new random with a fixed seed for debug consistancy
 * @author Jonathan Hudgins jhudgins8@gatech.edu
 */
public class RandomFactory {
    /** Random number generator */
    public static Random newRandom() {
        Random random = new Random();
        // comment/uncomment to get diverse/consistent behavior
        // random.setSeed(1234);
        return random;
    }
}
    
