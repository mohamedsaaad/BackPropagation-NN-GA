import java.util.ArrayList;

public class TrainExample {

    ArrayList<Double> inputs;
    ArrayList<Double> outputs;

    public TrainExample(ArrayList<Double> outputs, ArrayList<Double> inputs) {
        this.outputs = outputs;
        this.inputs = inputs;
    }

    public TrainExample() {
    }

    public ArrayList<Double>  getOutputs() {
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
}
