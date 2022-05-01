package nnassignment;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;

public class NNAssignment {


    public static void main(String[] args) throws FileNotFoundException {
       FeedForward obj=new FeedForward("D:\\fci\\sanarab3a\\genetics\\Assignments\\Assignment 3\\train.txt");
        obj.readFromFile();  
        obj.intialWeights();
        obj.calculateOutPutOfNural();
        obj.culculateMSE();
//        System.out.println("the out put"+obj.getOutFromOutPutLayer());
//        System.out.println("numOfInputLayer: "+obj.getNumOfInputLayer());
//        System.out.println("numOfHiddenLayer: "+obj.getNumOfHiddenLayer());
//        System.out.println("numOfOutputLayer: "+obj.getNumOfOutputLayer());
//        System.out.println("numOfTrainingExample: "+obj.getNumOfTrainingExample());
//        for (int i = 0; i <obj.getInputRows().size(); i++) {
//            System.out.println(obj.getInputRows().get(i));
//        }
//        System.out.println("the output"+obj.actualOutPut);
//        System.out.println("the MSE: "+obj.mSEForExmples.size());
    }
}
