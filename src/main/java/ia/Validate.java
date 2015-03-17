package ia;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;

public class Validate {

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

    private static Instances trainSet;

    private static final int TRAIN_SET_SIZE = 3189;

    public static void main(String... args) throws Exception {
        Arrays.stream(attributeNames).map(Attribute::new).forEach(atts::addElement);

        readTrainningSet();

        MultilayerPerceptron net = createNet(trainSet);

        System.out.println("Creating evaluation");
        Evaluation evaluation = new Evaluation(trainSet);
        System.out.println("Starting evaluation");
        evaluation.crossValidateModel(net, trainSet, 10, new Debug.Random(1));
        System.out.println(evaluation.toSummaryString());
    }

    private static MultilayerPerceptron createNet(Instances set) throws Exception {
        MultilayerPerceptron net = new MultilayerPerceptron();

        net.setHiddenLayers("30,30,30");
        net.setTrainingTime(150);
        net.setLearningRate(0.1d);
        System.out.println("Building net");
        net.buildClassifier(set);
        return net;
    }

    private static void readTrainningSet() throws IOException {
        trainSet = new Instances("trainning-set", atts, TRAIN_SET_SIZE);
        trainSet.setClass(trainSet.attribute(trainSet.numAttributes() - 1));
        Files.lines(Paths.get("data/trainning.csv")).map(Validate::line2instance).forEach(trainSet::add);
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
