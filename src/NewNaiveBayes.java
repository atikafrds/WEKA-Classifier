import weka.classifiers.AbstractClassifier;
import weka.core.Instances;
import weka.core.Instance;

/* A Naive Bayes Model Classifier
 * Limitations:
 * - Can only accept nominal data
 * - every data must have values
 */

public class NewNaiveBayes extends AbstractClassifier {
	/* Create a model table.
	 * First array for attributes,
	 * Second array for class values
	 * Third array for it's value
	   This table is a frequency table. The probs is used when classifying
	 */
	int[][][] nbTable;
	
	// Create a holder for class frequency
	int[] classFreq;
	
	// Sizes & stuff
	int classIndex = 0;
	int sizeOfClassVal = 0;
	int sizeOfAtts = 0;
	int sizeOfAttVal[];
	int sizeOfInst = 0;
	
	
	// Constructor
	public NewNaiveBayes() {
	}
	
	// Build Classifier method
	@Override
	public void buildClassifier(Instances arg0) throws Exception {
                int i,j,k;
		// Usable variables
                // Num of values of classes in arg0
                sizeOfClassVal = arg0.attribute(arg0.classIndex()).numValues();

                // Class index
                classIndex = arg0.classIndex();

                // Num of instances
                sizeOfInst = arg0.numInstances();

                // Num of attributes
                sizeOfAtts = arg0.numAttributes();
                        
		// Initialize model
		sizeOfAttVal = new int[sizeOfAtts];
		classFreq = new int[sizeOfClassVal];
			
		// find max number of values in attributes
		int maxAttVal = 0;
		for (i=0;i<sizeOfAtts;i++) {
			sizeOfAttVal[i] = arg0.attribute(i).numValues();
			if (maxAttVal < sizeOfAttVal[i]) {
				maxAttVal = sizeOfAttVal[i];
			}
		}
		
		// Initialize table
		nbTable = new int[sizeOfAtts][sizeOfClassVal][maxAttVal];
		for (i=0;i<sizeOfClassVal;i++) {
			classFreq[i] = 0;
			for (j=0;j<sizeOfAtts;j++) {
				for (k=0;k<maxAttVal;k++) {
					nbTable[j][i][k] = 0;
				}
			}
		}
		
		// Iterate through instances to fill the table
		for(i=0;i<sizeOfInst;i++) {
			int classVal = (int)arg0.instance(i).classValue();
	
			// Save frequency of class values
			classFreq[classVal]++;
			
			// Iterate through attributes to save frequency of its value according to classes
			//TODO: If an attributes doesn't have a value, there's a problem
			for (j=0;j<sizeOfAtts;j++) {
				// the value is assumed as integer (nominal)
				int attVal = (int) arg0.instance(i).value(j);
				nbTable[j][classVal][attVal]++;
			}
		}
                
                
                double[][][] probTable = new double[sizeOfAtts][sizeOfClassVal][maxAttVal];
                // Initialize table
		for (i=0;i<sizeOfClassVal;i++) {
                    for (j=0;j<sizeOfAtts;j++) {
                        for (k=0;k<maxAttVal;k++) {
                            probTable[j][i][k] = 0.0;
                        }
                    }
		}
                 // Calculate the probs for each class values
		for (i=0;i<sizeOfClassVal;i++) {
			// Iterate through attributes
			for (j=0;j<sizeOfAtts;j++) {
                            for (k=0;k<maxAttVal;k++) {
				// If the attribute is not Class Attribute, calculate the Pi function of P(a_i|v_i)
				if (j!=classIndex) {
                                    probTable[j][i][k] = 1.0 * nbTable[j][i][k] / classFreq[i];
				}
                            }
			}
		}
                
                for (i = 0; i < sizeOfInst; i++) {
                    classifyInstance(arg0.instance(i));
                }
	}
		
	// Classify Data method
        @Override
	public double classifyInstance(Instance arg1) {
		int i,j;
		double probs;
		double maxProbs = 0;
		int maxInd = 0;
		// Calculate the probs for each class values
		for (i=0;i<sizeOfClassVal;i++) {
        		double probAtt = 1;
			// Iterate through attributes
			for (j=0;j<arg1.numAttributes();j++) {
				// If the attribute is not Class Attribute, calculate the Pi function of P(a_i|v_i)
				if (j!=classIndex) {
                                    probAtt = probAtt * nbTable[j][i][(int)arg1.value(j)] / classFreq[i];
				}
			}
			
			// Calculate the P(v_i) * Pi-function (from above)
			probs = probAtt * classFreq[i]/sizeOfInst;
			// Compare the maximum probs
			if (maxProbs < probs) {
				maxProbs = probs;
				maxInd = i;
			}
		}
		
		// Return the index of maximum probability
		return (double) maxInd;
	}
}
