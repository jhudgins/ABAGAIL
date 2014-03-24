package util;

import java.io.PrintWriter;
import java.text.DecimalFormat;

import shared.DataSet;
import shared.Instance;
import util.linalg.Matrix;
import util.linalg.Vector;

/**
 * A utility for preparing and presenting run time metrics.
 * 
 * @author Jonathan Hudgins jhudgins8@gatech.edu
 * @date 2014-03-21
 */
public class PythonOut {
    public static String prefix;
    private static PrintWriter writer;
    private static DecimalFormat scientificFormat  = new DecimalFormat("0.000E00");
    private static DecimalFormat longFormat        = new DecimalFormat("0.0000000");
    private static DecimalFormat normalFormat      = new DecimalFormat("0.00000");

    public static void startFile(String name, String file) {
        if (writer != null) {
            writer.close();
        }
        try {
        	writer = new PrintWriter(file);
        } catch (Exception e) {
        	System.out.println(e);
        }
        prefix = name;
    }

    public static String format(double value) {
        double absValue = Math.abs(value);
        if (absValue == 0.0) {
            return normalFormat.format(value);
        } else if (absValue > 1000 || absValue < 0.0001) {
            return scientificFormat.format(value);
        } else if (absValue < 0.01) {
            return longFormat.format(value);
        } else {
            return normalFormat.format(value);
        }
    }

    public static void write(double value) {
        writer.print(format(value));
    }

    public static void write(String str) {
        writer.print(str);
        writer.flush();
    }

    public static void write(String name, double d) {
        writer.println(prefix + "['" + name + "'] = " + d);
        writer.flush();
    }

    public static void write(String name, int i) {
        writer.println(prefix + "['" + name + "'] = " + i);
        writer.flush();
    }

    public static void write(String name, double[] d) {
        writer.print(prefix + "['" + name + "'] = (");
        for (int i=0; i<d.length; i++) {
            write(d[i]);
            writer.print(",");
        }
        writer.println(")");
        writer.flush();
    }

    public static void write(String name, int[] iarray) {
        writer.print(prefix + "['" + name + "'] = (");
        for (int i=0; i<iarray.length; i++) {
            writer.print(iarray[i]);
            writer.print(",");
        }
        writer.println(")");
        writer.flush();
    }

    public static void write(String name, Vector v) {
        writer.print(prefix + "['" + name + "'] = (");
        for (int i=0; i<v.size(); i++) {
            write(v.get(i));
            writer.print(",");
        }
        writer.println(")");
        writer.flush();
    }

    public static void write(String name, Matrix m) {
        writer.println(prefix + "['" + name + "'] = (");
        for (int i=0; i<m.m(); i++) {
            writer.print("  (");
            for (int j=0; j<m.n(); j++) {
                write(m.get(i, j));
                writer.print(",");
            }
            writer.println("),");
        }
        writer.println(")");
        writer.flush();
    }

    public static void write(String name, DataSet d) {
        writer.println(prefix + "['" + name + "'] = (");
        Instance[] instances = d.getInstances();
        for (int i=0; i<instances.length; i++) {
            Instance instance = instances[i];
            writer.print("  (");
            for (int j=0; j<instance.size(); j++) {
                write(instance.getContinuous(j));
                writer.print(",");
            }
            writer.print(instance.getLabel().getDiscrete());
            writer.print(",");
            writer.println("),");
        }
        writer.println(")");
        writer.flush();
    }
}
