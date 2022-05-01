import java.io.*;
import java.util.ArrayList;
import java.util.Scanner;

public class NNAssignment {
    static int numOfTrainingExample;
    static double dataMax,datMin;
    static ArrayList<TrainExample> examples =new ArrayList<>();
    static NeuralNetwork network=new NeuralNetwork();
    public NNAssignment() {
    }

    public int getNumOfTrainingExample() {
        return numOfTrainingExample;
    }
    public void setNumOfTrainingExample(int numOfTrainingExample) {
        this.numOfTrainingExample = numOfTrainingExample;
    }
    public static ArrayList<TrainExample> getExamples() {
        return examples;
    }
    public static void setExamples(ArrayList<TrainExample> examples) {
        NNAssignment.examples = examples;
    }
    public static NeuralNetwork getNetwork() {
        return network;
    }
    public static void setNetwork(NeuralNetwork network) {
        NNAssignment.network = network;
    }

    //read data from file
    public static void readFromFile() throws FileNotFoundException {
        File text = new File("D:\\\\Education\\\\FCI\\\\fourth year\\\\genetics algorithim\\\\assignment-4\\\\NN\\\\train.txt");
        Scanner scnr = new Scanner(text);
        int numOfInputLayer=scnr.nextInt();
        int numOfHiddenLayer=scnr.nextInt();
        int numOfOutputLayer=scnr.nextInt();
        network.setNumOfInputs(numOfInputLayer);
        network.setNumOfHidden(numOfHiddenLayer);
        network.setNumOfOutput(numOfOutputLayer);
        numOfTrainingExample = scnr.nextInt();
        for (int j = 0; j < numOfTrainingExample; j++) {
            ArrayList<Double> row = new ArrayList<>();
            ArrayList<Double> rowOutPut = new ArrayList<>();
            TrainExample trainExample=new TrainExample();
            for (int i = 0; i < numOfInputLayer; i++) {
                double temp = scnr.nextDouble();
                row.add(temp);
            }
            trainExample.setInputs(row);
            for (int i = 0; i < numOfOutputLayer; i++) {
                double temp = scnr.nextDouble();
                rowOutPut.add(temp);
            }
            trainExample.setOutputs(rowOutPut);
            examples.add(trainExample);
        }
        normalizeData();
    }

    //get min and max of data
    public static void getMaxandMin(){
        double max=Double.MIN_VALUE,min=Double.MAX_VALUE;
        for (int i = 0; i < examples.size(); i++) {
            for (int j = 0; j < examples.get(i).getInputs().size(); j++) {
                if (examples.get(i).getInputs().get(j)>= max)
                    max=examples.get(i).getInputs().get(j);
                if (examples.get(i).getInputs().get(j)<= min)
                    min=examples.get(i).getInputs().get(j);
            }
            for (int j = 0; j < examples.get(i).getOutputs().size(); j++) {
                if (examples.get(i).getOutputs().get(j)>= max)
                    max=examples.get(i).getOutputs().get(j);
                if (examples.get(i).getOutputs().get(j)<= min)
                    min=examples.get(i).getOutputs().get(j);
            }
        }
        dataMax=max;
        datMin=min;
    }

    //normalize data
    public static void normalizeData(){
        getMaxandMin();
        for (int i = 0; i < examples.size(); i++) {
            for (int j = 0; j < examples.get(i).getInputs().size(); j++) {
                examples.get(i).getInputs().set(j,(examples.get(i).getInputs().get(j)-datMin)/(dataMax-datMin));
            }
            for (int j = 0; j < examples.get(i).getOutputs().size(); j++) {
                examples.get(i).getOutputs().set(j,(examples.get(i).getOutputs().get(j)-datMin)/(dataMax-datMin));
            }
        }
    }

    //save weights and MSE after learning
    public static void saveLearningData() throws FileNotFoundException {
        PrintStream out=new PrintStream(new File("learningData.txt"));
        out.println(network.getMeanSquareError());
        for (int i = 0; i < network.getWeightsFromInToHid().size(); i++) {
            for (int j = 0; j < network.getWeightsFromInToHid().get(i).size(); j++) {
                out.print(network.getWeightsFromInToHid().get(i).get(j)+" ");
            }
        }
        out.println();
        for (int i = 0; i < network.getWeightsFromHidToOut().size(); i++) {
            for (int j = 0; j < network.getWeightsFromHidToOut().get(i).size(); j++) {
                out.print(network.getWeightsFromHidToOut().get(i).get(j)+" ");
            }
        }
        out.println();
        out.close();
    }

    //running backPropagation learning algorithm
    public static void runBackProLearningALgo() throws FileNotFoundException {
        readFromFile();
        network.intiateWeights();
        double meanSE=0;
        for (int i = 0; i < 300; i++) {
            meanSE=0;
            for (int j = 0; j < numOfTrainingExample; j++) {
                network.fetchExample(examples.get(j));
                meanSE+=network.feedForwarding();
                network.packPropagating();
            }
            meanSE=meanSE/numOfTrainingExample;
//            System.out.println("before:"+meanSE);
            meanSE=(meanSE*(dataMax-datMin))+datMin;
//            System.out.println("After: "+meanSE);
            if(meanSE>=0.001 && meanSE<=0.002)
                network.setMeanSquareError(meanSE);
        }
        network.setMeanSquareError(meanSE);
        saveLearningData();
    }

    //read learning data from file
    public static void readLearningData() throws FileNotFoundException {
        File x= new File("learningData.txt");
        Scanner in=new Scanner(x);
        network.setMeanSquareError(in.nextDouble());
        ArrayList<ArrayList<Double>> weightsFromInToHid=new ArrayList<>();
        ArrayList<ArrayList<Double>> weightsFromHidToOut=new ArrayList<>();
        for (int i = 0; i < network.getNumOfHidden(); i++) {
            ArrayList<Double> row=new ArrayList<>();
            for (int j = 0; j < network.getNumOfInputs(); j++) {
                row.add(in.nextDouble());
            }
            weightsFromInToHid.add(row);
        }
        for (int i = 0; i < network.getNumOfOutput(); i++) {
            ArrayList<Double> row=new ArrayList<>();
            for (int j = 0; j < network.getNumOfHidden(); j++) {
                row.add(in.nextDouble());
            }
            weightsFromHidToOut.add(row);
        }
        network.setWeightsFromInToHid(weightsFromInToHid);
        network.setWeightsFromHidToOut(weightsFromHidToOut);
        in.close();
    }

    //running forward for network after learning
    public static void runForwardTest() throws FileNotFoundException {
        network=new NeuralNetwork();
        readFromFile();
        readLearningData();
        double meanSE=0;
        for (int j = 0; j < numOfTrainingExample; j++) {
            network.fetchExample(examples.get(j));
            meanSE+=network.feedForwarding();
        }
        meanSE=meanSE/numOfTrainingExample;
        System.out.println("learned MSE: "+network.getMeanSquareError());
        System.out.println("current MSE: "+meanSE);
    }


    public static void main(String[] args) throws FileNotFoundException {
        runBackProLearningALgo();
        runForwardTest();
    }
}
