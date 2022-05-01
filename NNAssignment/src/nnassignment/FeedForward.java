package nnassignment;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

public class FeedForward {

    int numOfInputLayer;
    int numOfHiddenLayer;
    int numOfOutputLayer;
    int numOfTrainingExample;
    String fileName;
    ArrayList<ArrayList<Double>> inputRows = new ArrayList<>();
    ArrayList<ArrayList<Double>> weightsOfHiddenLayer = new ArrayList<>();
    ArrayList<ArrayList<Double>> weightsOfOutPut = new ArrayList<>();
    ArrayList<ArrayList<Double>> actualOutPut = new ArrayList<>();
    ArrayList<ArrayList<Double>> outFromHiddenLayer = new ArrayList<>();
    ArrayList<ArrayList<Double>> outFromOutPutLayer = new ArrayList<>();
    ArrayList<Double> mSEForExmples = new ArrayList<>();

    public FeedForward() {
    }

    public FeedForward(String fileName) {
        this.fileName = fileName;
    }

    public FeedForward(int numOfInputLayer, int numOfHiddenLayer, int numOfOutputLayer, String fileName) {
        this.numOfInputLayer = numOfInputLayer;
        this.numOfHiddenLayer = numOfHiddenLayer;
        this.numOfOutputLayer = numOfOutputLayer;
        this.fileName = fileName;
    }

    public int getNumOfInputLayer() {
        return numOfInputLayer;
    }

    public void setNumOfInputLayer(int numOfInputLayer) {
        this.numOfInputLayer = numOfInputLayer;
    }

    public int getNumOfHiddenLayer() {
        return numOfHiddenLayer;
    }

    public void setNumOfHiddenLayer(int numOfHiddenLayer) {
        this.numOfHiddenLayer = numOfHiddenLayer;
    }

    public int getNumOfOutputLayer() {
        return numOfOutputLayer;
    }

    public void setNumOfOutputLayer(int numOfOutputLayer) {
        this.numOfOutputLayer = numOfOutputLayer;
    }

    public int getNumOfTrainingExample() {
        return numOfTrainingExample;
    }

    public void setNumOfTrainingExample(int numOfTrainingExample) {
        this.numOfTrainingExample = numOfTrainingExample;
    }

    public String getFileName() {
        return fileName;
    }

    public void setFileName(String fileName) {
        this.fileName = fileName;
    }

    public ArrayList<ArrayList<Double>> getInputRows() {
        return inputRows;
    }

    public void setInputRows(ArrayList<ArrayList<Double>> inputRows) {
        this.inputRows = inputRows;
    }

    public ArrayList<ArrayList<Double>> getWeightsOfHiddenLayer() {
        return weightsOfHiddenLayer;
    }

    public void setWeightsOfHiddenLayer(ArrayList<ArrayList<Double>> weightsOfHiddenLayer) {
        this.weightsOfHiddenLayer = weightsOfHiddenLayer;
    }

    public ArrayList<ArrayList<Double>> getWeightsOfOutPut() {
        return weightsOfOutPut;
    }

    public void setWeightsOfOutPut(ArrayList<ArrayList<Double>> weightsOfOutPut) {
        this.weightsOfOutPut = weightsOfOutPut;
    }

    public ArrayList<ArrayList<Double>> getActualOutPut() {
        return actualOutPut;
    }

    public void setActualOutPut(ArrayList<ArrayList<Double>> actualOutPut) {
        this.actualOutPut = actualOutPut;
    }

    public ArrayList<ArrayList<Double>> getOutFromHiddenLayer() {
        return outFromHiddenLayer;
    }

    public void setOutFromHiddenLayer(ArrayList<ArrayList<Double>> outFromHiddenLayer) {
        this.outFromHiddenLayer = outFromHiddenLayer;
    }

    public ArrayList<ArrayList<Double>> getOutFromOutPutLayer() {
        return outFromOutPutLayer;
    }

    public void setOutFromOutPutLayer(ArrayList<ArrayList<Double>> outFromOutPutLayer) {
        this.outFromOutPutLayer = outFromOutPutLayer;
    }

// read data from file
    public void readFromFile() throws FileNotFoundException {
        File text = new File(this.fileName);
        Scanner scnr = new Scanner(text);
        this.numOfInputLayer = scnr.nextInt();
        numOfHiddenLayer = scnr.nextInt();
        numOfOutputLayer = scnr.nextInt();
        numOfTrainingExample = scnr.nextInt();
        for (int j = 0; j < numOfTrainingExample; j++) {
            ArrayList<Double> row = new ArrayList<>();
            ArrayList<Double> rowOutPut = new ArrayList<>();
            for (int i = 0; i < numOfInputLayer; i++) {
                double temp = scnr.nextDouble();
                row.add(temp);
            }
            inputRows.add(row);
            for (int i = 0; i < numOfOutputLayer; i++) {
                double temp = scnr.nextDouble();
                rowOutPut.add(temp);
            }
            actualOutPut.add(rowOutPut);
        }
    }

    // get intial weights for the network
    public void intialWeights() {
        //iniate matrix of weights with length L * M for hidden layer
        for (int i = 0; i < this.numOfHiddenLayer; i++) {
            ArrayList<Double> row = new ArrayList<>();
            for (int j = 0; j < this.numOfInputLayer; j++) {
                Random r = new Random();
                double randomValue = -10 + (10 - (-10)) * r.nextDouble();
                row.add(randomValue);
            }
            this.weightsOfHiddenLayer.add(row);
        }
        //iniate matrix of weights with length N * L for outPut Layer
        for (int i = 0; i < this.numOfOutputLayer; i++) {
            ArrayList<Double> row = new ArrayList<>();
            for (int j = 0; j < this.numOfHiddenLayer; j++) {
                Random r = new Random();
                double randomValue = -10 + (10 - (-10)) * r.nextDouble();
                row.add(randomValue);
            }
            this.weightsOfOutPut.add(row);
        }
    }
    //calculate output from hidden layer & output layer for evry training example

    public void calculateOutPutOfNural() {
        for (int k = 0; k < numOfTrainingExample; k++) {
            ArrayList<Double> outFromHiddenForK = new ArrayList<>();
            ArrayList<Double> outFromOutPutForK = new ArrayList<>();
            for (int i = 0; i < this.weightsOfHiddenLayer.size(); i++) {
                double temp = 0;
                for (int j = 0; j < this.weightsOfHiddenLayer.get(i).size(); j++) {
                    temp = temp + weightsOfHiddenLayer.get(i).get(j) * inputRows.get(k).get(j);
                }
                outFromHiddenForK.add(temp);
            }
            outFromHiddenLayer.add(outFromHiddenForK);
            for (int i = 0; i < weightsOfOutPut.size(); i++) {
                double num = 0;
                for (int j = 0; j < weightsOfOutPut.get(i).size(); j++) {
                    num = num + weightsOfOutPut.get(i).get(j) * outFromHiddenForK.get(j);
                }
                outFromOutPutForK.add(num);
            }
            outFromOutPutLayer.add(outFromOutPutForK);
        }
    }

    // the sigmoid Function
    public double activationSigmoid(Double out) {
        double result = 1 / 1 + (Math.pow(Math.E, 0 - out));
        return result;
    }

    public void culculateMSE() {
        for (int i = 0; i < actualOutPut.size(); i++) {
            double mSE = 0;
            for (int j = 0; j < actualOutPut.get(i).size(); j++) {
                double diffrence = actualOutPut.get(i).get(j) - outFromOutPutLayer.get(i).get(j);
                mSE += Math.pow(diffrence, 2);
            }
            mSEForExmples.add(0.5*mSE);
        }
    }
}
