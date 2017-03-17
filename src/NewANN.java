import static java.lang.Math.exp;
import java.util.Random;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.classifiers.RandomizableClassifier;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;
//import weka.filters.unsupervised.attribute
/* An ANN Model Classifier
 * Limitations:
 * - Can only accept numeric data
 * - every data must have values
 * - Can only have exactly 1 hidden layer (can be 0)
 */

public class NewANN extends AbstractClassifier {
    int sizeOfInput = 0;
    int sizeOfHidden = 0;
    int sizeOfOutput = 0;
    int sizeOfHLNeuron = 0;
    int numInstances = 0;
    int maxIter = 0;
    double learningRate = 0.001;
    double[] HLNeuron;
    double[] output;
    double[] input;

    /* If the hidden layer is 0, then weightIH[][] will be used
    as matrix to save the weight between input and output.
    And weightHO[][] will not be used. */
    double[][] weightIH;
    double[][] weightHO;

    // Constructor
    public NewANN(int hidden, int neuron, int max) {
        sizeOfHidden = hidden;
        sizeOfHLNeuron = neuron;
        maxIter = max;
    }

    @Override
    public void buildClassifier(Instances inputData) throws Exception {
        
    	// Initialize Variables
        sizeOfInput = inputData.numAttributes() - 1;
        if (inputData.attribute(inputData.classIndex()).numValues() == 2) {
            sizeOfOutput = 1;
        } else {
            sizeOfOutput = inputData.attribute(inputData.classIndex()).numValues();
        }
        numInstances = inputData.numInstances();
        
        // Initialize Sizes
        input = new double[sizeOfInput+1];
        output = new double[sizeOfOutput];
        System.out.println("Size of input = " + sizeOfInput);
        System.out.println("Size of output = " + sizeOfOutput);
        System.out.println("Num of Instances = " + numInstances);
        System.out.println("Size of hidden layer = " + sizeOfHidden);
        
        Random rand = new Random(0);
        
        // Divide between 1 hidden and no hidden
        if (sizeOfHidden == 1) {
            // Fill with random number
            // Random initial weight + bias weight
            weightIH = new double[sizeOfInput+1][sizeOfHLNeuron];
            for (int j = 0; j < sizeOfInput+1; j++) {
                for (int k = 0; k < sizeOfHLNeuron; k++) {
                    weightIH[j][k] = -0.2 + (0.4* rand.nextDouble());
                }
            }
            weightHO = new double[sizeOfHLNeuron][sizeOfOutput];
            for (int i = 0; i < sizeOfHLNeuron; i++) {
                for (int j = 0; j < sizeOfOutput; j++) {
                    weightHO[i][j] = -0.2 + (0.4* rand.nextDouble());
                }
            }
            HLNeuron = new double[sizeOfHLNeuron];

            // Main process
            double err;
            int iterate = 1;
            do {
//            	System.out.println("Iteration: " + iterate);
            	iterate++;
                err = 0;
                for (int i = 0; i < numInstances; i++) {
                    // Calculate ANN
                	output = ANN_Hidden(inputData.instance(i));

                    // Back Propagation for hidden-output
                    double[] errorOutput = new double[sizeOfOutput];
                    for (int j = 0; j < sizeOfOutput; j++) {
                        double target = 0.0;
                        if (sizeOfOutput == 1) {
                            target = inputData.instance(i).classValue();
                        } else {
                            if (inputData.instance(i).classValue() == (double)j) {
                                target = 1.0;
                            }
                        }
                        errorOutput[j] = output[j] * (1 - output[j]) * (target - output[j]);
                        for (int k = 0; k < sizeOfHLNeuron; k++) {
                            weightHO[k][j] = weightHO[k][j] + (learningRate * errorOutput[j] * HLNeuron[k]);
                        }
                    }

                    // Back Propagation for input-hidden
                    for (int j = 0; j < sizeOfHLNeuron; j++) {
                        double backPropragate = 0;
                        for (int k = 0; k < sizeOfOutput; k++) {
                            backPropragate = backPropragate + (weightHO[j][k] * errorOutput[k]);
                        }
                        double error = HLNeuron[j] * (1 - HLNeuron[j]) * backPropragate;
                        for (int m = 0; m < sizeOfInput+1; m++) {
                            weightIH[m][j] = weightIH[m][j] + (learningRate * error * input[m]);
                        }
                    }
                }
                
                // Errors
                int erri = 0;
                for (int i = 0; i < numInstances; i++) {
                	double out = classifyInstance(inputData.instance(i));
                    err = err + Math.pow(inputData.instance(i).classValue()-out,2);

                    if (inputData.instance(i).classValue()!=out) {
                    	erri++;
                    }
                }
                err = err/numInstances;
                System.out.println(err);
//                System.out.println("NBE: " + erri);
                
            } while ((iterate < maxIter) && (err > 1));

        // sizeOfHLNeuron = 0
        } else {
        	// Randomize the weight
            weightIH = new double[sizeOfInput+1][sizeOfOutput];
            for (int i = 0; i < sizeOfInput+1; i++) {
                for (int j = 0; j < sizeOfOutput; j++) {
                    weightIH[i][j] = rand.nextDouble();
                }
            }

            // Main process
            double err;
            int iterate = 1;
            do {
                err = 0;
//            	System.out.println("Iteration: " + iterate);
            	iterate++;
                for (int i = 0; i < numInstances; i++) {
                    // Calculate ANN
                	output = ANN_noHidden(inputData.instance(i));

                    // Back Propagation
                    for (int j = 0; j < sizeOfOutput; j++) {
                        double target = 0.0;
                        if (sizeOfOutput == 1) {
                            target = inputData.instance(i).classValue();
                        } else {
                            if (inputData.instance(i).classValue() == (double)j) {
                                target = 1.0;
                            }
                        }
                        
                        for (int k = 0; k < sizeOfInput+1; k++) {
                            weightIH[k][j] = weightIH[k][j] + (learningRate * output[j] * (1 - output[j]) * (target - output[j]) * input[k]);
                        }
                    }
                }
                
                // Errors
                int erri = 0;
                for (int i = 0;i<numInstances;i++) {
                	// Classify the instance
                	double out = classifyInstance(inputData.instance(i));
                    err = err + Math.pow(inputData.instance(i).classValue()-out,2);
                    
                    if (inputData.instance(i).classValue()!=out) {
                    	erri++;
                    }
                }
                err = err/numInstances;
            } while ((iterate < maxIter) && (err > 1));
                
        }
    }

    @Override
    public double classifyInstance(Instance arg0) {
        double out = 0;
    	// divide between 1 hidden and no hidden
    	if (sizeOfHidden == 1) {
    		// Calculate ANN
    		output = ANN_Hidden(arg0);
            
            // Classify the instance
            if (sizeOfOutput == 1) {
            	if (output[0]>=0.5) {
            		out = 1;
            	} else {
            		out = 0;
            	}
            } else {
            	double maxout = 0;
            	for (int j = 0; j<sizeOfOutput;j++) {
            		if (maxout<output[j]) {
            			maxout = output[j];
            			out = (double) j;
            		}
            	}
            }
    	} else {
    		// Calculate ANN
    		output = ANN_noHidden(arg0);
            
            // Classify the instance
            if (sizeOfOutput == 1) {
            	if (output[0]>=0.5) {
            		out = 1;
            	} else {
            		out = 0;
            	}
            } else {
            	double maxout = 0;
            	for (int j = 0; j<sizeOfOutput;j++) {
            		if (maxout<output[j]) {
            			maxout = output[j];
            			out = (double) j;
            		}
            	}
            }
    	}
        // return
        return out;
    }
        
    public double sigmoid(double arg0) {
            return (1/(1 + Math.exp(-arg0)));
    }
    
    public double[] ANN_Hidden(Instance arg0) {
    	// Initialize input
        input[0] = 1;
        int index = 0;
        for (int j = 1; j < sizeOfInput+1; j++) {
            if (index == arg0.classIndex()) {
                index++;
            }
            input[j] = arg0.value(index);
            index++;
        }
        
        // Calculate the hidden
        for (int j = 0; j < sizeOfHLNeuron; j++) {
            double sigma = 0;
            for (int k = 0; k < sizeOfInput+1; k++) {
                sigma = sigma + (input[k] * weightIH[k][j]);                            
            }
            HLNeuron[j] = sigmoid(sigma);
        }
        
        // Calculate output
        for (int j = 0; j < sizeOfOutput; j++) {
            double sigma = 0;
            for (int k = 0; k < sizeOfHLNeuron; k++) {
                sigma = sigma + (HLNeuron[k] * weightHO[k][j]);
            }
            output[j] = sigmoid(sigma);
        }
        
        return output;
    }
    
    public double[] ANN_noHidden(Instance arg0) {
    	// initialize input
        input[0] = 1;
        int index = 0;
        for (int j = 1; j < sizeOfInput+1; j++) {
            if (index == arg0.classIndex()) {
                index++;
            }
            input[j] = arg0.value(index);
            index++;
        }
        
        // calculate the output through ANN
        for (int j = 0; j < sizeOfOutput; j++) {
            double sigma = 0;
            for (int k = 0; k < sizeOfInput+1; k++) {
                sigma = sigma + (input[k] * weightIH[k][j]);
            }
            output[j] = sigmoid(sigma);
        }
        
        return output;
    }
}