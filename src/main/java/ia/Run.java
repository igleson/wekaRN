package ia;

import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.*;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Scanner;

public class Run {

    private static String[] attributeNames = new String[]{
            "acidez_fixa ",
            "acidez_volatil",
            "acido_citrico",
            "acucar_residual",
            "cloretos",
            "dioxido_enxofre_livre",
            "densidade",
            "pH",
            "sulfatos",
            "alcool"};

    private static FastVector atts = new FastVector();
    private static Instances testSet;
    private static Instances trainSet;
    private static Instances useSet;

    private static final int TRAIN_SET_SIZE = 3189;
    private static final int TEST_SET_SIZE = 1470;

    private static final double THRESHOLD = 0.47;

    public static void main(String... args) throws Exception {
        Arrays.stream(attributeNames).map(Attribute::new).forEach(atts::addElement);

        readTrainningSet();
        readTestSet();
        createUserSet();

        BufferedWriter w = new BufferedWriter(new FileWriter(new File("data/output.dat")));

        MultilayerPerceptron net = createNet(trainSet);

        for (int i = 0; i < testSet.numInstances(); i++) {
//            double res = net.classifyInstance(testSet.instance(i));
//            int clazz = 0;
//            if(res > 0.47){
//                clazz = 1;
//            }
//            w.write(clazz + "\n");
        }

        w.close();

        Scanner sc = new Scanner(System.in);
        while (true) {
            System.out.println("You wish to classify a instance?(y/n)");
            String cont = sc.next();
            if (!cont.equalsIgnoreCase("y")) {
                break;
            }
            System.out.println("Type your entry:");

            String in = sc.next();
            Instance instance = line2instance(in);
            useSet.add(instance);
            instance.setDataset(useSet);

            Double out = net.classifyInstance(instance);
            if (out < THRESHOLD) {
                System.out.println("Quality: bad");
            } else {
                System.out.println("Quality: good");
            }
        }
    }

    private static MultilayerPerceptron createNet(Instances set) throws Exception {
        MultilayerPerceptron net = new MultilayerPerceptron();

//        net.setHiddenLayers("30,30,30");
//        net.setTrainingTime(150);
//        net.setLearningRate(0.1d);
        System.out.println("Building net");
        net.buildClassifier(set);
        return net;
    }

    private static void readTestSet() throws IOException {
        System.out.println("Reading test set");
        testSet = new Instances("trainning-set", atts, TEST_SET_SIZE);
        testSet.setClass(testSet.attribute(testSet.numAttributes() - 1));

        Files.lines(Paths.get("data/test.csv")).map(Run::line2instance).forEach(inst -> {
            inst.setDataset(testSet);
            testSet.add(inst);
        });
        System.out.println("Done reading test set");
    }

    private static void createUserSet() throws IOException {
        System.out.println("Creating use set");
        useSet = new Instances("use-set", atts, 50);
        useSet.setClass(useSet.attribute(useSet.numAttributes() - 1));

        System.out.println("Done creating use set");
    }

    private static void readTrainningSet() throws IOException {
        trainSet = new Instances("trainning-set", atts, TRAIN_SET_SIZE);
        trainSet.setClass(trainSet.attribute(trainSet.numAttributes() - 1));
        Files.lines(Paths.get("data/trainning.csv")).map(Run::line2instance).forEach(trainSet::add);
    }

    private static Instance line2instance(String line) {
        String[] vals = line.split(",");
        double[] parsedVals = new double[vals.length];
        for (int i = 0; i < vals.length; i++) {
            parsedVals[i] = Double.parseDouble(vals[i]);
        }
        return new SparseInstance(1.0, parsedVals);
    }
}
