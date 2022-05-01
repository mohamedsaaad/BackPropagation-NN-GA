import java.util.ArrayList;
import java.util.Random;

public class NeuralNetwork {
    int numOfInputs;
    int numOfHidden;
    int numOfOutput;
    double learningRate=0.1;
    double meanSquareError;
    ArrayList<Double> inputs;
    ArrayList<Double> hiddens;
    ArrayList<Double> outputs;
    ArrayList<Double> targetOutputs;
    ArrayList<ArrayList<Double>> weightsFromInToHid;
    ArrayList<ArrayList<Double>> weightsFromHidToOut;

    public NeuralNetwork() {
        ArrayList<Double> inputs=new ArrayList<>();
        ArrayList<Double> hiddens=new ArrayList<>();
        ArrayList<Double> outputs=new ArrayList<>();
        ArrayList<Double> targetOutputs=new ArrayList<>();
        ArrayList<ArrayList<Double>> weightsFromInToHid=new ArrayList<>();
        ArrayList<ArrayList<Double>> weightsFromHidToOut=new ArrayList<>();
    }

    public int getNumOfInputs() {
        return numOfInputs;
    }
    public void setNumOfInputs(int numOfInputs) {
        this.numOfInputs = numOfInputs;
    }
    public int getNumOfHidden() {
        return numOfHidden;
    }
    public void setNumOfHidden(int numOfHidden) {
        this.numOfHidden = numOfHidden;
    }
    public int getNumOfOutput() {
        return numOfOutput;
    }
    public void setNumOfOutput(int numOfOutput) {
        this.numOfOutput = numOfOutput;
    }
    public ArrayList<Double> getHiddens() {
        return hiddens;
    }
    public void setHiddens(ArrayList<Double> hiddens) {
        this.hiddens = hiddens;
    }
    public ArrayList<Double> getOutputs() {
        return outputs;
    }
    public void setOutputs(ArrayList<Double> outputs) {
        this.outputs = outputs;
    }
    public ArrayList<Double> getInputs() {
        return inputs;
    }
    public void setInputs(ArrayList<Double> inputs) {
        this.inputs = inputs;
    }
    public ArrayList<ArrayList<Double>> getWeightsFromInToHid() {
        return weightsFromInToHid;
    }
    public void setWeightsFromInToHid(ArrayList<ArrayList<Double>> weightsFromInToHid) {
        this.weightsFromInToHid = weightsFromInToHid;
    }
    public ArrayList<ArrayList<Double>> getWeightsFromHidToOut() {
        return weightsFromHidToOut;
    }
    public void setWeightsFromHidToOut(ArrayList<ArrayList<Double>> weightsFromHidToOut) {
        this.weightsFromHidToOut = weightsFromHidToOut;
    }
    public ArrayList<Double> getActualOutputs() {
        return targetOutputs;
    }
    public void setActualOutputs(ArrayList<Double> actualOutputs) {
        targetOutputs = actualOutputs;
    }
    public double getLearningRate() {
        return learningRate;
    }
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
    public ArrayList<Double> getTargetOutputs() {
        return targetOutputs;
    }
    public void setTargetOutputs(ArrayList<Double> targetOutputs) {
        this.targetOutputs = targetOutputs;
    }
    public double getMeanSquareError() {
        return meanSquareError;
    }
    public void setMeanSquareError(double meanSquareError) {
        this.meanSquareError = meanSquareError;
    }

    //initiate random weights to the network
    public void intiateWeights() {
        this.weightsFromInToHid=new ArrayList<>();
        this.weightsFromHidToOut=new ArrayList<>();
        //intiate matrix of weights with length L * M for hidden layer
        for (int i = 0; i < numOfHidden; i++) {
            ArrayList<Double> row = new ArrayList<>();
            for (int j = 0; j < numOfInputs; j++) {
                Random r = new Random();
                double randomValue = -10 + (10 - (-10)) * r.nextDouble();
                row.add(randomValue);
            }
            this.weightsFromInToHid.add(row);
        }
        //intiate matrix of weights with length N * L for outPut Layer
        for (int i = 0; i < numOfOutput; i++) {
            ArrayList<Double> row = new ArrayList<>();
            for (int j = 0; j < numOfHidden; j++) {
                Random r = new Random();
                double randomValue = -10 + (10 - (-10)) * r.nextDouble();
                row.add(randomValue);
            }
            this.weightsFromHidToOut.add(row);
        }
    }

    // the sigmoid Function
    public double activationSigmoid(Double out) {
        double result = 1 / (1 + (Math.pow(Math.E, 0 - out)));
        return result;
    }

    //calculate output from hidden layer & output layer for evry training example
    public void calculateOutPut() {
        hiddens=new ArrayList<>();
        outputs=new ArrayList<>();
        for (int i = 0; i < weightsFromInToHid.size(); i++) {
            double temp = 0;
            for (int j = 0; j < weightsFromInToHid.get(i).size(); j++) {
                temp = temp + weightsFromInToHid.get(i).get(j) * inputs.get(j);
            }

//            System.out.println(temp+" -> "+activationSigmoid(temp));
            hiddens.add(activationSigmoid(temp));
        }
        for (int i = 0; i < weightsFromHidToOut.size(); i++) {
            double num = 0;
            for (int j = 0; j < weightsFromHidToOut.get(i).size(); j++) {
                num = num + weightsFromHidToOut.get(i).get(j) * hiddens.get(j);
            }
            outputs.add(activationSigmoid(num));
        }
    }

    //fetch train example ana fill inputs
    public void fetchExample(TrainExample trainExample)
    {
        inputs=new ArrayList<>();
        targetOutputs=new ArrayList<>();
        for (int i = 0; i < numOfInputs; i++)
            inputs.add(trainExample.getInputs().get(i));
        for (int i = 0; i < numOfOutput; i++)
            targetOutputs.add(trainExample.getOutputs().get(i));
    }

    //calculate MSE for network
    public double calculateMSE() {
        double mSE = 0;
        for (int i = 0; i < numOfOutput; i++) {
            double diffrence = targetOutputs.get(i) - outputs.get(i);
            mSE += Math.pow(diffrence, 2);
        }
        return mSE;
    }

    //apply feed forward algorithm to the network
    public double feedForwarding()
    {
        calculateOutPut();
         return calculateMSE();
    }

    //calculate output layer errors
    public ArrayList<Double> calcOutError()
    {
        ArrayList<Double> errors=new ArrayList<>();
        double error=0;
        for (int i = 0; i < numOfOutput ; i++) {
            error=outputs.get(i)*(1-outputs.get(i))*(targetOutputs.get(i)-outputs.get(i));
        }
        errors.add(error);
        return errors;
    }

    //calculate new weights between hidden and output layer
    public ArrayList<ArrayList<Double>> calcNewWeightFromHidToOut(ArrayList<Double> outErrors)
    {
        ArrayList<ArrayList<Double>> newWeights=new ArrayList<>();
        for (int i = 0; i < numOfOutput; i++) {
            ArrayList<Double> weightsByOutput=new ArrayList<>();
            double newWeight=0;
            for (int j = 0; j < numOfHidden; j++) {
                newWeight=weightsFromHidToOut.get(i).get(j)+learningRate*outErrors.get(i)*hiddens.get(j);
                weightsByOutput.add(newWeight);
            }
            newWeights.add(weightsByOutput);
        }
        return newWeights;
    }

    //calculate hidden layer errors
    public ArrayList<Double> calcHiddenError(ArrayList<Double> outErrors)
    {
        ArrayList<Double> errors=new ArrayList<>();
        double error=0;
        for (int i = 0; i < numOfHidden ; i++) {
            double sum=0;
            for (int j=0 ; j < numOfOutput ; j++)
            {
                sum+=outErrors.get(j)*weightsFromHidToOut.get(j).get(i);
            }
            error=hiddens.get(i)*(1-hiddens.get(i))+sum;
            errors.add(error);
        }
        return errors;
    }

    //calculate new weights between input and hidden layer
    public ArrayList<ArrayList<Double>> calcNewWeightFromInToHiD(ArrayList<Double> hiddenErrors)
    {
        ArrayList<ArrayList<Double>> newWeights=new ArrayList<>();
        for (int i = 0; i < numOfHidden; i++) {
            ArrayList<Double> weightsByHidden=new ArrayList<>();
            double newWeight=0;
            for (int j = 0; j < numOfInputs; j++) {
                newWeight=weightsFromInToHid.get(i).get(j)+learningRate*hiddenErrors.get(i)*inputs.get(j);
                weightsByHidden.add(newWeight);
            }
            newWeights.add(weightsByHidden);
        }
        return newWeights;
    }

    //apply back propagation algorithm to the network
    public void packPropagating()
    {
        ArrayList<Double> outErrors=calcOutError();
        ArrayList<ArrayList<Double>> newWeightsFromHidToOut=calcNewWeightFromHidToOut(outErrors);
        ArrayList<Double> hiddenErrors=calcHiddenError(outErrors);
        ArrayList<ArrayList<Double>> newWeightsFromInToHid=calcNewWeightFromInToHiD(hiddenErrors);
        setWeightsFromHidToOut(newWeightsFromHidToOut);
        setWeightsFromInToHid(newWeightsFromInToHid);
    }

}
