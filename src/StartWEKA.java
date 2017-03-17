import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.ObjectOutputStream;
import java.util.Random;
import weka.classifiers.Classifier;

import weka.classifiers.evaluation.*;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;

public class StartWEKA {
	
	public static void main(String[] args) throws Exception {
		
		// Read the arff file
		BufferedReader breader ;
		breader = new BufferedReader(new FileReader("student-train.arff"));
		
		// Create instances (datums)
		Instances train1 = new Instances (breader);
                
                // Close the file
		breader.close();

		train1.setClassIndex(27);
                
//                NominalToBinary filter = new NominalToBinary();
//                filter.setInputFormat(train1);
//                Instances train2 = Filter.useFilter(train1, filter);
//                
//                Normalize filter2 = new Normalize();
//                filter2.setInputFormat(train2);
//                Instances inst = Filter.useFilter(train2, filter2);
                
//                inst.randomize(new java.util.Random(1));		
//                int trainSize = (int) Math.round(inst.numInstances() * 0.8);
//                int testSize = inst.numInstances() - trainSize;
//                Instances train = new Instances(inst, 0, trainSize);
//                Instances test = new Instances(inst, trainSize, testSize);
                
//                Discretize filter = new Discretize();
//                filter.setInputFormat(train1);
//                Instances train = Filter.useFilter(train1, filter);
		
                // Naive Bayes Classification
//		NewNaiveBayes nB = new NewNaiveBayes();
		
		//Insert the training data to nB
//		nB.buildClassifier(train);
		
//		Evaluation eval = new Evaluation(train);
//		eval.evaluateModel(nB, train);
//		eval.crossValidateModel(nB, train, 10, new Random());
//                System.out.println(eval.toSummaryString("=== Summary ===", true));
//                System.out.println(eval.toClassDetailsString("=== Detailed Accuracy By Class ==="));
//                System.out.println(eval.toMatrixString("=== Confusion matrix ==="));

//		System.out.println(eval.toSummaryString("\nResults\n=====\n", true));
//		System.out.println(eval.fMeasure(1) + " "+ eval.precision(1) + " " + eval.recall(1));
//              weka.core.SerializationHelper.write("student_walc_ffnn_split.model", ann);
                
              NewANN annmodel = (NewANN) weka.core.SerializationHelper.read("Team.model");
              System.out.println("\nReading model done\n");
              breader = new BufferedReader(new FileReader("Team_test.arff"));
              Instances test1 = new Instances(breader);
              test1.setClassIndex(test1.numAttributes() - 1);
              
              NominalToBinary filter = new NominalToBinary();
            filter.setInputFormat(test1);
            Instances train2 = Filter.useFilter(test1, filter);
            
            Normalize filter2 = new Normalize();
            filter2.setInputFormat(train2);
            Instances test = Filter.useFilter(train2, filter2);
            
//            Discretize filter = new Discretize();
//                filter.setInputFormat(test1);
//                Instances test = Filter.useFilter(test1, filter);
            
            Evaluation eval = new Evaluation(test);
              eval.evaluateModel(annmodel, test);
              System.out.println(eval.toSummaryString("=== Summary ===", true));
              System.out.println(eval.toClassDetailsString("=== Detailed Accuracy By Class ==="));
              System.out.println(eval.toMatrixString("=== Confusion matrix ==="));
//NewNaiveBayes nB = (NewNaiveBayes) weka.core.SerializationHelper.read("mush_1_split test_99,6308.model");
//              
//            breader = new BufferedReader(new FileReader("mush_test.arff"));
//	 Instances test1 = new Instances (breader);
//         test1.setClassIndex(0);
            
//                filter = new NominalToBinary();
//                filter.setInputFormat(test);
//                train2 = Filter.useFilter(test, filter);
//                
//                filter2 = new Normalize();
//                filter2.setInputFormat(train2);
//                test = Filter.useFilter(train2, filter2);
                
//                Discretize filter = new Discretize();
//                filter.setInputFormat(test1);
//                Instances test = Filter.useFilter(test1, filter);
//
//                    Evaluation eval = new Evaluation(test);
//                eval.evaluateModel(nB, test);
//            System.out.println(eval.toSummaryString("=== Summary ===", true));
//            System.out.println(eval.toClassDetailsString("=== Detailed Accuracy By Class ==="));
//            System.out.println(eval.toMatrixString("=== Confusion matrix ==="));
//            System.out.println(eval.fMeasure(1) + " " + eval.recall(1));
	}
}
