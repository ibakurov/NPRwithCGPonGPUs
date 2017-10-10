package main;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import javax.imageio.ImageIO;

import ec.EvolutionState;
import ec.Individual;
import ec.cgp.Evaluator;
import ec.cgp.Record;
import ec.cgp.problems.ClassificationProblem;
import ec.cgp.representation.VectorIndividualCGP;
import ec.cgp.representation.VectorSpeciesCGP;
import ec.gp.cuda.CudaInterop;
import ec.gp.cuda.KernelOutputData;
import ec.multiobjective.SumOfRanksFitness;
import ec.util.Parameter;
import fitness.DeviationFromNormal;
import fitness.HistogramDistance;
import fitness.Luminance;
import fitness.Mean;
import fitness.StandardDeviation;
import jcuda.runtime.JCuda;

public class NPRWithCGPOnGPUsProblem extends ClassificationProblem {

	private static final long serialVersionUID = 1755954437542996060L;
	
	private int numberOfBrushstrokesToBeAppliedToOneIndividual = 100;
	private int numberOfBrushstrokesToBeAppliedToOneIndividualWithBigImage = 500;
	private int percentageOfPixelsWhichShouldBePaintedOnOneIndividual = 80;
	private int percentageOfPixelsWhichShouldBePaintedOnOneIndividualWithBigImage = 80;
	private int numberOfPaintedPixels = 0;
    FileWriter fw;
    BufferedWriter bw;
    PrintWriter out;
	protected static int imageNumber = 0;

	private boolean useCuda = true;
	
    public static PixelRecord currentPixel;
    
	/** It's here so as to prevent excessive casting */
	public ProblemData problemData = new ProblemData();

	//--------
    //These fields are calculated in this class, but used as functions in language of CGP
    
    //Lum of source image, because it doesn't change, we calculate it only once
    public static float luminance = (float) 0.0;
    
    //We recalculate further fields after each brushstroke applied. We calculate them based on canvas image
    
    //5x5
    //Mean
    public static float mean5x5 = (float) 0.0;
    //STD
    public static float std5x5 = (float) 0.0;
    //Min
    public static float min5x5 = (float) 0.0;
    //Max
    public static float max5x5 = (float) 0.0;
    
    //7x7
    //Mean
    public static float mean7x7 = (float) 0.0;
    //STD
    public static float std7x7 = (float) 0.0;
    //Min
    public static float min7x7 = (float) 0.0;
    //Max
    public static float max7x7 = (float) 0.0;
    
    //9x9
    //Mean
    public static float mean9x9 = (float) 0.0;
    //STD
    public static float std9x9 = (float) 0.0;
    //Min
    public static float min9x9 = (float) 0.0;
    //Max
    public static float max9x9 = (float) 0.0;

    //11x11
    //Mean
    public static float mean11x11 = (float) 0.0;
    //STD
    public static float std11x11 = (float) 0.0;
    //Min
    public static float min11x11 = (float) 0.0;
    //Max
    public static float max11x11 = (float) 0.0;

    //13x13
    //Mean
    public static float mean13x13 = (float) 0.0;
    //STD
    public static float std13x13 = (float) 0.0;
    //Min
    public static float min13x13 = (float) 0.0;
    //Max
    public static float max13x13 = (float) 0.0;
    
	static float ERC = new Random().nextFloat();
    
    //--------
	
    public void setup(final EvolutionState state, final Parameter base) {
    	super.setup(state, base);
    	Evaluator.setup(state, this);
    	CudaInterop.getInstance().setup(state);
    	CudaInterop.getInstance().prepareKernel(state);
//		JCuda.cudaDeviceSynchronize();
		
    	setupBufferWriters();
    	// Now initialize the problem data arrays
    	this.problemData.init();

		calculateLuminanceValueForGraphLanguage(false);
    }
    
    public void setupBufferWriters() {
    	try {
			fw = new FileWriter("../Results/Images/" + ProblemData.titleOfFolder + "/outfilename" + ProblemData.titleOfFolder + ".txt", true);
			bw = new BufferedWriter(fw);
			out = new PrintWriter(bw);
		} catch (IOException e) {
			e.printStackTrace();
		}
    }
    
    /** 
	 * Obtain the inputs from the data record.
	 */
	protected void setInputs(Object[] inputs, Record rec) {
		if (rec.getClass().equals(PixelRecord.class)) {
			PixelRecord r = (PixelRecord) rec;
			inputs[0] = (float) r.x;
			inputs[1] = (float) r.y;
			inputs[2] = (float) (r.redInput / 255.0);
			inputs[3] = (float) (r.greenInput / 255.0);
			inputs[4] = (float) (r.blueInput / 255.0);
			inputs[5] = (float) (r.redCanvas / 255.0);
			inputs[6] = (float) (r.greenCanvas / 255.0);
			inputs[7] = (float) (r.blueCanvas / 255.0);
			inputs[8] = (float) r.opacity;
			inputs[9] = luminance;
			inputs[10] = ERC;
			inputs[11] = mean5x5;
			inputs[12] = mean7x7;
			inputs[13] = mean9x9;
			inputs[14] = mean11x11;
			inputs[15] = mean13x13;
			inputs[16] = std5x5;
			inputs[17] = std7x7;
			inputs[18] = std9x9;
			inputs[19] = std11x11;
			inputs[20] = std13x13;
			inputs[21] = min5x5;
			inputs[22] = min7x7;
			inputs[23] = min9x9;
			inputs[24] = min11x11;
			inputs[25] = min13x13;
			inputs[26] = max5x5;
			inputs[27] = max7x7;
			inputs[28] = max9x9;
			inputs[29] = max11x11;
			inputs[30] = max13x13;
		}
	}
	
	/** 
	 * Obtain the inputs from the data record.
	 */
	Float[] getInputs(Record rec, int numInputs) {
		Float[] inputs = new Float[numInputs];
		if (rec.getClass().equals(PixelRecord.class)) {
			PixelRecord r = (PixelRecord) rec;
			inputs[0] = (float) r.x;
			inputs[1] = (float) r.y;
			inputs[2] = (float) (r.redInput / 255.0);
			inputs[3] = (float) (r.greenInput / 255.0);
			inputs[4] = (float) (r.blueInput / 255.0);
			inputs[5] = (float) (r.redCanvas / 255.0);
			inputs[6] = (float) (r.greenCanvas / 255.0);
			inputs[7] = (float) (r.blueCanvas / 255.0);
			inputs[8] = (float) r.opacity;
			inputs[9] = luminance;
			inputs[10] = ERC;
			inputs[11] = mean5x5;
			inputs[12] = mean7x7;
			inputs[13] = mean9x9;
			inputs[14] = mean11x11;
			inputs[15] = mean13x13;
			inputs[16] = std5x5;
			inputs[17] = std7x7;
			inputs[18] = std9x9;
			inputs[19] = std11x11;
			inputs[20] = std13x13;
			inputs[21] = min5x5;
			inputs[22] = min7x7;
			inputs[23] = min9x9;
			inputs[24] = min11x11;
			inputs[25] = min13x13;
			inputs[26] = max5x5;
			inputs[27] = max7x7;
			inputs[28] = max9x9;
			inputs[29] = max11x11;
			inputs[30] = max13x13;
		}
		return inputs;
	}
        
    public void evaluate(final EvolutionState state, final Individual ind, final int subpopulation, final int threadnum) {
    	if (ind.evaluated)	// don't bother reevaluating
    		return;
    	
    	//Reset image for the next individual
    	//Fully black
        this.problemData.canvasImage = new BufferedImage(this.problemData.inputImage.getWidth(), this.problemData.inputImage.getHeight(), BufferedImage.TYPE_INT_RGB);
        //Fully white
    	for (int i = 0; i < problemData.canvasImage.getWidth(); i++) {
        	for (int j = 0; j < problemData.canvasImage.getHeight(); j++) {
        		int col = (255 << 16) | (255 << 8) | 255;
        		problemData.canvasImage.setRGB(i, j, col);
        		((PixelRecord)this.problemData.data.get(i).get(j)).haveBeenColoured = false;
        	}
    	}	
    	numberOfPaintedPixels = 0;
    	
    	VectorSpeciesCGP s = (VectorSpeciesCGP) ind.species;
		VectorIndividualCGP ind2 = (VectorIndividualCGP) ind;
		
		
//		while (numberOfPaintedPixels < this.problemData.data.size()*this.problemData.data.get(0).size()*percentageOfPixelsWhichShouldBePaintedOnOneIndividual/100) {
		for (int i = 0; i < numberOfBrushstrokesToBeAppliedToOneIndividual; i++) {
			int randomPixelX = new Random().nextInt(this.problemData.data.size());
			int randomPixelY = new Random().nextInt(this.problemData.data.get(randomPixelX).size());
			PixelRecord pixel = (PixelRecord)this.problemData.data.get(randomPixelX).get(randomPixelY);
			PixelRecord finalPixel = null;
			if (!pixel.haveBeenColoured) {
				//Now if we haven't process this pixel yet, we are going to apply brush stroke with it.
				finalPixel = pixel;				
			} else {
				//If we have already processed this pixel, we will go through pixels in the image and apply brushstroke to the next unprocessed pixel.
				boolean didFindNewPixel = false;
				for (int k = randomPixelY; k < problemData.inputImage.getHeight(); k++) {
					for (int l = 0; l < problemData.inputImage.getWidth(); l++) {
						PixelRecord newPixel = (PixelRecord)this.problemData.data.get(l).get(k);
						if (!pixel.haveBeenColoured) {
							finalPixel = newPixel;	
							didFindNewPixel = true;
							break;
						}
					}
				}
				//If we got to the end of the image and still didn't find any unprocessed pixel, we will start from the start and will be going till randomly generated one
				if (!didFindNewPixel) {
					for (int k = 0; k <= randomPixelY; k++) {
						for (int l = 0; l < problemData.inputImage.getWidth(); l++) {
							PixelRecord newPixel = (PixelRecord)this.problemData.data.get(l).get(k);
							if (!pixel.haveBeenColoured) {
								finalPixel = newPixel;	
								didFindNewPixel = true;
								break;
							}
						}
					}
				}
				//Finally, if we didn't find any new unprocessed pixel, let's just go with the randomized one
				if (!didFindNewPixel) {
					finalPixel = pixel;
				}
			}
			
			currentPixel = finalPixel;
		
			calculateAllNeccessaryFieldsForGraphLanguage(5, finalPixel, false);
			calculateAllNeccessaryFieldsForGraphLanguage(7, finalPixel, false);
			calculateAllNeccessaryFieldsForGraphLanguage(9, finalPixel, false);
			calculateAllNeccessaryFieldsForGraphLanguage(11, finalPixel, false);
			calculateAllNeccessaryFieldsForGraphLanguage(13, finalPixel, false);
			
			Float[] inputs = getInputs(finalPixel, s.numInputs);
			Float[] results = null;
			
			results = eval(state, threadnum, inputs, ind2);
			float scale = Math.abs(results[0]);
			float rotation = results[1];
			int bitmapNumber = results[2].intValue();
			
			applyBrushstroke(state, ind2, threadnum, bitmapNumber, scale, rotation, finalPixel, problemData.data, problemData.canvasImage, false);
		}
		
		//Assign fitness based on CanvasImage
			double MEAN = new Mean(0.0).calculateFitness(null, this.problemData.canvasImage, this.problemData.inputImage)[1];
			double SD = new StandardDeviation(0.0).calculateFitness(null, this.problemData.canvasImage, this.problemData.inputImage)[1];
			
			double DFN = new DeviationFromNormal(0).calculateFitness(null, this.problemData.canvasImage, this.problemData.inputImage)[1];

			double CHISTQ = new HistogramDistance(0).calculateFitness(null, this.problemData.canvasImage, this.problemData.targetCHISTQImage)[1];
			double LUM = new Luminance().distance(new Luminance().turnImageToGrayScale(this.problemData.canvasImage), new Luminance().turnImageToGrayScale(this.problemData.inputImage));
			
			SumOfRanksFitness f = ((SumOfRanksFitness)ind.fitness);
			f.setObjectives(state, new double[]
					{
							Math.abs(3.2 - MEAN),
							Math.abs(0.75 - SD),
							Math.abs(0 - DFN),
							CHISTQ,
							LUM
					});
			ind2.evaluated = true;
		    out.println(Math.abs(3.2 - MEAN) + " " + Math.abs(0.75 - SD) + " " + Math.abs(0 - DFN) + " " + CHISTQ + " " + LUM + " " + ind2.expression);
			ind2.expression.append("MEAN: " + MEAN + "SD:" + SD + "; DFN: " + DFN + "; CHISTQ: " + CHISTQ + "; LUM: " + LUM);
		    
			DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyyMMdd HH:mm:ss");
			LocalDateTime now = LocalDateTime.now();
			
			// Save image
	    	File outputfile = new File("../Results/Images/" + ProblemData.titleOfFolder + "/saved" + imageNumber + "-" + dtf.format(now) + ".png");
	    	try {
	    		ImageIO.write(problemData.canvasImage, "png", outputfile);
	    		imageNumber++;
	    	} catch (IOException e) {
	    		e.printStackTrace();
	    	}
    }
    
    /**
	 * Use the best individual of the run and apply it to the test image
	 * To see the results performance image is created
	 */
	@Override
	public void describe(EvolutionState state, Individual ind, int subpopulation, int threadnum, int log) {
		super.describe(state, ind, subpopulation, threadnum, log);
						
		//-------
		//First: Describe the best individual from all the run!
		//-------
		
		VectorSpeciesCGP s = (VectorSpeciesCGP) ind.species;
		VectorIndividualCGP ind2 = (VectorIndividualCGP) ind;

		//Reset image for the next individual
    	//Fully black
        this.problemData.canvasImage = new BufferedImage(this.problemData.inputImage.getWidth(), this.problemData.inputImage.getHeight(), BufferedImage.TYPE_INT_RGB);
        //Fully white
    	for (int i = 0; i < problemData.canvasImage.getWidth(); i++) {
        	for (int j = 0; j < problemData.canvasImage.getHeight(); j++) {
        		int col = (255 << 16) | (255 << 8) | 255;
        		problemData.canvasImage. setRGB(i, j, col);
        		((PixelRecord)this.problemData.data.get(i).get(j)).haveBeenColoured = false;
        	}
    	}
    	numberOfPaintedPixels = 0;
    	
//		while (numberOfPaintedPixels < this.problemData.data.size()*this.problemData.data.get(0).size()*percentageOfPixelsWhichShouldBePaintedOnOneIndividual/100) {
    	for (int i = 0; i < numberOfBrushstrokesToBeAppliedToOneIndividual; i++) {
			int randomPixelX = new Random().nextInt(this.problemData.data.size());
			int randomPixelY = new Random().nextInt(this.problemData.data.get(randomPixelX).size());
			PixelRecord pixel = (PixelRecord)this.problemData.data.get(randomPixelX).get(randomPixelY);
			PixelRecord finalPixel = null;
			if (!pixel.haveBeenColoured) {
				//Now if we haven't process this pixel yet, we are going to apply brush stroke with it.
				finalPixel = pixel;				
			} else {
				//If we have already processed this pixel, we will go through pixels in the image and apply brushstroke to the next unprocessed pixel.
				boolean didFindNewPixel = false;
				for (int k = randomPixelY; k < problemData.inputImage.getHeight(); k++) {
					for (int l = 0; l < problemData.inputImage.getWidth(); l++) {
						PixelRecord newPixel = (PixelRecord)this.problemData.data.get(l).get(k);
						if (!pixel.haveBeenColoured) {
							finalPixel = newPixel;	
							didFindNewPixel = true;
							break;
						}
					}
				}
				//If we got to the end of the image and still didn't find any unprocessed pixel, we will start from the start and will be going till randomly generated one
				if (!didFindNewPixel) {
					for (int k = 0; k <= randomPixelY; k++) {
						for (int l = 0; l < problemData.inputImage.getWidth(); l++) {
							PixelRecord newPixel = (PixelRecord)this.problemData.data.get(l).get(k);
							if (!pixel.haveBeenColoured) {
								finalPixel = newPixel;	
								didFindNewPixel = true;
								break;
							}
						}
					}
				}
				//Finally, if we didn't find any new unprocessed pixel, let's just go with the randomized one
				if (!didFindNewPixel) {
					finalPixel = pixel;
				}
			}
			
			currentPixel = finalPixel;
			
			calculateAllNeccessaryFieldsForGraphLanguage(5, finalPixel, false);
			calculateAllNeccessaryFieldsForGraphLanguage(7, finalPixel, false);
			calculateAllNeccessaryFieldsForGraphLanguage(9, finalPixel, false);
			calculateAllNeccessaryFieldsForGraphLanguage(11, finalPixel, false);
			calculateAllNeccessaryFieldsForGraphLanguage(13, finalPixel, false);

			Float[] inputs = getInputs(finalPixel, s.numInputs);
			Float[] results = null;
			
			results = eval(state, threadnum, inputs, ind2);
			float scale = Math.abs(results[0]);
			float rotation = results[1];
			int bitmapNumber = results[2].intValue();
	
			applyBrushstroke(state, ind2, threadnum, bitmapNumber, scale, rotation, finalPixel, problemData.data, problemData.canvasImage, false);
		}
//		}
		
		DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyyMMdd HH:mm:ss");
		LocalDateTime now = LocalDateTime.now();
		
		// Save image
    	File outputfile = new File("../Results/Images/" + ProblemData.titleOfFolder + "/savedBest" + imageNumber + "-" + dtf.format(now) + ".png");
    	try {
    		ImageIO.write(problemData.canvasImage, "png", outputfile);
    		imageNumber++;
    	} catch (IOException e) {
    		e.printStackTrace();
    	}
    	
    	//-------
		//Second: Describe the best individual from all the run on a big image!
		//-------
    	
		calculateLuminanceValueForGraphLanguage(true);
		
		numberOfPaintedPixels = 0;
		
//		while (numberOfPaintedPixels < this.problemData.dataBig.size()*this.problemData.dataBig.get(0).size()*percentageOfPixelsWhichShouldBePaintedOnOneIndividualWithBigImage/100) {
		for (int i = 0; i < numberOfBrushstrokesToBeAppliedToOneIndividualWithBigImage; i++) {
			int randomPixelX = new Random().nextInt(this.problemData.dataBig.size());
			int randomPixelY = new Random().nextInt(this.problemData.dataBig.get(randomPixelX).size());
			PixelRecord pixel = (PixelRecord)this.problemData.dataBig.get(randomPixelX).get(randomPixelY);
			PixelRecord finalPixel = null;
			if (!pixel.haveBeenColoured) {
				//Now if we haven't process this pixel yet, we are going to apply brush stroke with it.
				finalPixel = pixel;				
			} else {
				//If we have already processed this pixel, we will go through pixels in the image and apply brushstroke to the next unprocessed pixel.
				boolean didFindNewPixel = false;
				for (int k = randomPixelY; k < problemData.inputBigImage.getHeight(); k++) {
					for (int l = 0; l < problemData.inputBigImage.getWidth(); l++) {
						PixelRecord newPixel = (PixelRecord)this.problemData.dataBig.get(l).get(k);
						if (!pixel.haveBeenColoured) {
							finalPixel = newPixel;	
							didFindNewPixel = true;
							break;
						}
					}
				}
				//If we got to the end of the image and still didn't find any unprocessed pixel, we will start from the start and will be going till randomly generated one
				if (!didFindNewPixel) {
					for (int k = 0; k <= randomPixelY; k++) {
						for (int l = 0; l < problemData.inputBigImage.getWidth(); l++) {
							PixelRecord newPixel = (PixelRecord)this.problemData.dataBig.get(l).get(k);
							if (!pixel.haveBeenColoured) {
								finalPixel = newPixel;	
								didFindNewPixel = true;
								break;
							}
						}
					}
				}
				//Finally, if we didn't find any new unprocessed pixel, let's just go with the randomized one
				if (!didFindNewPixel) {
					finalPixel = pixel;
				}
			}
			
			currentPixel = finalPixel;
			
			calculateAllNeccessaryFieldsForGraphLanguage(5, finalPixel, true);
			calculateAllNeccessaryFieldsForGraphLanguage(7, finalPixel, true);
			calculateAllNeccessaryFieldsForGraphLanguage(9, finalPixel, true);
			calculateAllNeccessaryFieldsForGraphLanguage(11, finalPixel, true);
			calculateAllNeccessaryFieldsForGraphLanguage(13, finalPixel, true);

			Float[] inputs = getInputs(finalPixel, s.numInputs);
			Float[] results = null;
			
			results = eval(state, threadnum, inputs, ind2);
			float scale = Math.abs(results[0]);
			float rotation = results[1];
			int bitmapNumber = results[2].intValue();
	
			applyBrushstroke(state, ind2, threadnum, bitmapNumber, scale, rotation, finalPixel, problemData.dataBig, problemData.canvasBigImage, true);
		}
//		}
		
		//Assign fitness based on canvasBigImage
		double MEANBig = new Mean(0.0).calculateFitness(null, this.problemData.canvasBigImage, this.problemData.inputBigImage)[1];
		double SDBig = new StandardDeviation(0.0).calculateFitness(null, this.problemData.canvasBigImage, this.problemData.inputBigImage)[1];
		
		double DFNBig = new DeviationFromNormal(0).calculateFitness(null, this.problemData.canvasBigImage, this.problemData.inputBigImage)[1];
		double CHISTQBig = new HistogramDistance(0).calculateFitness(null, this.problemData.canvasBigImage, this.problemData.targetCHISTQBigImage)[1];
		double LUMBig = new Luminance().distance(new Luminance().turnImageToGrayScale(this.problemData.canvasBigImage), new Luminance().turnImageToGrayScale(this.problemData.inputBigImage));
		
		out.println("----------- FINAL RESULT - BIG IMAGE -------------");
		out.println(Math.abs(3.2 - MEANBig) + " " + Math.abs(0.75 - SDBig) + " " + Math.abs(0 - DFNBig) + " " + CHISTQBig + " " + LUMBig + " " + ind2.expression);
	    
		DateTimeFormatter dtfBig = DateTimeFormatter.ofPattern("yyyyMMdd HH:mm:ss");
		LocalDateTime nowBig = LocalDateTime.now();
		
		// Save image
    	File outputfileBig = new File("../Results/Images/" + ProblemData.titleOfFolder + "/BigImage-" + dtfBig.format(nowBig) + ".png");
    	try {
    		ImageIO.write(problemData.canvasBigImage, "png", outputfileBig);
    		imageNumber++;
    	} catch (IOException e) {
    		e.printStackTrace();
    	}
		    	
		out.close();
	}
	
	//---------
	//APPLICATION OF BRUSHSTROKE TO THE IMAGE
	//---------
    
    //Because I don't want to send these values to GPU and then retrieve them back, I am declaring them as a class variables, so that they are accessible later.
    int minimumX;
	int minimumY;
	int maximumX;
	int maximumY;
	
	
    public void applyBrushstroke(EvolutionState state, VectorIndividualCGP ind, int threadnum, 
			int bitmapNumber, float scale, float rotation, 
			PixelRecord pixel, List<ArrayList<Record>> data, BufferedImage canvasImage, boolean isItBigImage) {
		
    	//bitmap image we will apply as brush stroke
        BufferedImage bitmapImage = null;
		int imgW = isItBigImage ? this.problemData.inputBigImage.getWidth() : this.problemData.inputImage.getWidth();
		int imgH = isItBigImage ? this.problemData.inputBigImage.getHeight() : this.problemData.inputImage.getHeight();
		
		boolean shouldUseBigBrushstrokeInAnyWay = true;
		
	   //Read bitmap from the computer
        //Take mod of this value to the maximum possible value - 7
  		switch (Math.abs(bitmapNumber) % 7) {
  		case 0:
  	        try {
  	        	bitmapImage = (BufferedImage) ImageIO.read(new File("../Assets/Brushstrokes/0" + (shouldUseBigBrushstrokeInAnyWay ? "Big" : (isItBigImage ? "Big" : "")) + ".png"));
  			} catch (IOException e) {}
  			break;
  		case 1:
  	        try {
  	        	bitmapImage = (BufferedImage) ImageIO.read(new File("../Assets/Brushstrokes/1" + (shouldUseBigBrushstrokeInAnyWay ? "Big" : (isItBigImage ? "Big" : "")) + ".png"));
  			} catch (IOException e) {}
  			break;
  		case 2:
  	        try {
  	        	bitmapImage = (BufferedImage) ImageIO.read(new File("../Assets/Brushstrokes/2" + (shouldUseBigBrushstrokeInAnyWay ? "Big" : (isItBigImage ? "Big" : "")) + ".png"));
  			} catch (IOException e) {}
  			break;
  		case 3:
  	        try {
  	        	bitmapImage = (BufferedImage) ImageIO.read(new File("../Assets/Brushstrokes/3" + (shouldUseBigBrushstrokeInAnyWay ? "Big" : (isItBigImage ? "Big" : "")) + ".png"));
  			} catch (IOException e) {}
  			break;
  		case 4:
  	        try {
  	        	bitmapImage = (BufferedImage) ImageIO.read(new File("../Assets/Brushstrokes/4" + (shouldUseBigBrushstrokeInAnyWay ? "Big" : (isItBigImage ? "Big" : "")) + ".png"));
  			} catch (IOException e) {}
  			break;
  		case 5:
  	        try {
  	        	bitmapImage = (BufferedImage) ImageIO.read(new File("../Assets/Brushstrokes/5" + (isItBigImage ? "Big" : "") + ".png"));
  			} catch (IOException e) {}
  			break;
  		case 6:
  	        try {
  	        	bitmapImage = (BufferedImage) ImageIO.read(new File("../Assets/Brushstrokes/6" + (shouldUseBigBrushstrokeInAnyWay ? "Big" : (isItBigImage ? "Big" : "")) + ".png"));
  			} catch (IOException e) {}
  			break;
  		default:
  	        try {
  	        	bitmapImage = (BufferedImage) ImageIO.read(new File("../Assets/Brushstrokes/4" + (shouldUseBigBrushstrokeInAnyWay ? "Big" : (isItBigImage ? "Big" : "")) + ".png"));
  			} catch (IOException e) {}
  			break;
  		}
  		
  		//Get height width from the bitmap image
  		int originalBrushstrokeWidth = bitmapImage.getWidth();
  		int originalBrushstrokeHeight = bitmapImage.getHeight();
  		//These are the values which we allow for scale. Max an min values for scale. So we don't increase the size of brushstroke, and we don't make it too small.
  		float minScale = shouldUseBigBrushstrokeInAnyWay ? 0.35f : (isItBigImage ? 0.5f : 0.2f);
  		float maxScale = 1.0f;
  		
  		//Normalized scale is in between 0 and 1.
  		float normalizedScale = scale - (int)scale;
  		//Calcualte finalScale which lies in betwen 0.2 and 1.0 in regards of what CGP calucalted for us.
  		float finalScale = (float) (normalizedScale * (maxScale - minScale) + minScale);
  		
  		//Resize width and height of brushstroke according to scale
  		int resizedBrushstrokeWidth = (int) (originalBrushstrokeWidth * finalScale);
  		int resizedBrushstrokeHeight = (int) (originalBrushstrokeHeight * finalScale);

  		//Check that scale didn't break out of limits for brushstroke scaling.
  		if (resizedBrushstrokeWidth < originalBrushstrokeWidth * minScale) { resizedBrushstrokeWidth = (int) (originalBrushstrokeWidth * minScale); }
  		if (resizedBrushstrokeWidth > originalBrushstrokeWidth) { resizedBrushstrokeWidth = originalBrushstrokeWidth; }
  		if (resizedBrushstrokeHeight < originalBrushstrokeHeight * minScale) { resizedBrushstrokeHeight = (int) (originalBrushstrokeHeight * minScale); }
  		if (resizedBrushstrokeHeight > originalBrushstrokeHeight) { resizedBrushstrokeHeight = originalBrushstrokeHeight; }
        
  		//Create new resized Image buffer
  		Image tmp = bitmapImage.getScaledInstance(resizedBrushstrokeWidth, resizedBrushstrokeHeight, Image.SCALE_SMOOTH);
  	    BufferedImage bitmapImageScaled = new BufferedImage(resizedBrushstrokeWidth, resizedBrushstrokeHeight, BufferedImage.TYPE_INT_ARGB);
  	    Graphics2D g2d = bitmapImageScaled.createGraphics();
  	    g2d.drawImage(tmp, 0, 0, null);
  	    g2d.dispose();
  		
        //Extracting center (x, y) coordintates.
        int xCenter = pixel.x;
        int yCenter = pixel.y;
        
        //Calculating all (x, y) coordinates of corners. We calculate them, because in some cases they might be smaller then the width of the brushstroke.
        //Think about when brushstroke is applied in such way, that some part of it is out of the image.
        
        //The following is the ideal case:
        // 	-------------------
        //  |                 |
        //  |     -------     |
        //  |     |     |     |
        //  |     |  •  |     |
        //  |     |     |     |
        //  |     -------     |
        //  |                 |
        //  -------------------
        
        //The following is the case where brushstroke is a bit off the image:
        // 	-------------------
        //  |                 |
        //  |-----            |
        //  |    |            |
        //  | •  |            |
        //  |    |            |
        //  |-----            |
        //  |                 |
        //  -------------------
        
        //Or even this:
        // 	-------------------
        //  |                 |
        //  |                 |
        //  |                 |
        //  |                 |
        //  |             ----|
        //  |             |   |
        //  |             |  •|
        //  -------------------
        
        int topLeftXBeforeRotation = Math.max(0, xCenter - resizedBrushstrokeWidth / 2);
        int topLeftYBeforeRotation = Math.max(0, yCenter - resizedBrushstrokeHeight / 2);
        int topRightXBeforeRotation = Math.min(imgW-1, xCenter + resizedBrushstrokeWidth / 2);
        int topRightYBeforeRotation = Math.max(0, yCenter - resizedBrushstrokeHeight / 2);
        int botttomLeftXBeforeRotation = Math.max(0, xCenter - resizedBrushstrokeWidth / 2);
        int botttomLeftYBeforeRotation = Math.min(imgH-1, yCenter + resizedBrushstrokeHeight / 2);
        int bottomRightXBeforeRotation = Math.min(imgW-1, xCenter + resizedBrushstrokeWidth / 2);
        int bottomRightYBeforeRotation = Math.min(imgH-1, yCenter + resizedBrushstrokeHeight / 2);
        
        //Getting theta. Rotation in radians.
  		float theta = rotation;
        if (theta >= 360 || theta <= 0)	theta = 0;
        float thetaInRadians = (float) Math.toRadians(theta);

        //To rotate rectangle (brushstroke bitmap image) we use the following two formulas:
        // x' = xCenter + (x - xCenter)*cosØ + (y-yCenter)*sinØ
        // y' = yCenter - (x - xCenter)*sinØ + (y-yCenter)*cosØ
        //We apply these formulas to every x and y of every corner of the rectangle.
		int topLeftXAfterRotation = Math.min(imgW-1, Math.max(0, (int) Math.round((xCenter + (topLeftXBeforeRotation - xCenter)*Math.cos(thetaInRadians) + (topLeftYBeforeRotation - yCenter)*Math.sin(thetaInRadians)))));
		int topLeftYAfterRotation = Math.min(imgH-1, Math.max(0, (int) Math.round((yCenter - (topLeftXBeforeRotation - xCenter)*Math.sin(thetaInRadians) + (topLeftYBeforeRotation - yCenter)*Math.cos(thetaInRadians)))));
		int topRightXAfterRotation = Math.max(0, Math.min(imgW-1, (int) Math.round((xCenter + (topRightXBeforeRotation - xCenter)*Math.cos(thetaInRadians) + (topRightYBeforeRotation - yCenter)*Math.sin(thetaInRadians)))));
		int topRightYAfterRotation = Math.min(imgH-1, Math.max(0, (int) Math.round((yCenter - (topRightXBeforeRotation - xCenter)*Math.sin(thetaInRadians) + (topRightYBeforeRotation - yCenter)*Math.cos(thetaInRadians)))));
		int bottomLeftXAfterRotation = Math.min(imgW-1, Math.max(0, (int) Math.round((xCenter + (botttomLeftXBeforeRotation - xCenter)*Math.cos(thetaInRadians) + (botttomLeftYBeforeRotation - yCenter)*Math.sin(thetaInRadians)))));
		int bottomLeftYAfterRotation = Math.max(0, Math.min(imgH-1, (int) Math.round((yCenter - (botttomLeftXBeforeRotation - xCenter)*Math.sin(thetaInRadians) + (botttomLeftYBeforeRotation - yCenter)*Math.cos(thetaInRadians)))));
		int bottomRightXAfterRotation = Math.max(0, Math.min(imgW-1, (int) Math.round((xCenter + (bottomRightXBeforeRotation - xCenter)*Math.cos(thetaInRadians) + (bottomRightYBeforeRotation - yCenter)*Math.sin(thetaInRadians)))));
		int bottomRightYAfterRotation = Math.max(0, Math.min(imgH-1, (int) Math.round((yCenter - (bottomRightXBeforeRotation - xCenter)*Math.sin(thetaInRadians) + (bottomRightYBeforeRotation - yCenter)*Math.cos(thetaInRadians)))));
		
		//Ok, now we are looking for maximum and minimum (x, y) coordinates.
		//We do this, so that we could include our rectangle (brushstroke) into a square.
		//Minimum (x, y) is the coord of top left corner of a bigger square, which hovers the whole brushstroke
		//Maximum (x, y) is the coord of bottom right corner of a bigger square, which hovers the whole brushstroke
		minimumX = Math.min(Math.min(topLeftXAfterRotation, topRightXAfterRotation), Math.min(bottomLeftXAfterRotation, bottomRightXAfterRotation));
		minimumY = Math.min(Math.min(topLeftYAfterRotation, topRightYAfterRotation), Math.min(bottomLeftYAfterRotation, bottomRightYAfterRotation));
		maximumX = Math.max(Math.max(topLeftXAfterRotation, topRightXAfterRotation), Math.max(bottomLeftXAfterRotation, bottomRightXAfterRotation));
		maximumY = Math.max(Math.max(topLeftYAfterRotation, topRightYAfterRotation), Math.max(bottomLeftYAfterRotation, bottomRightYAfterRotation));
    	
		//Check if after all transforms on brushstroke it's still a valid brushstroke which we can apply to visible part of image.
		if (maximumX - minimumX != 0 && maximumY - minimumY != 0) {
		
	    	if (useCuda) {
				
				//This is very important!
				//Assign problem size the area of the square which holds the brushstroke
				int problemSize = (maximumX - minimumX) * (maximumY - minimumY);
				
//				System.out.println(problemSize + " " + maximumX + " " + minimumX + " " + maximumY + " " + minimumY  + " " + topLeftXBeforeRotation  + " " + topLeftYBeforeRotation  + " " + 
//						topRightXBeforeRotation + " " + topRightYBeforeRotation  + " " + botttomLeftXBeforeRotation  
//						 + " " + botttomLeftYBeforeRotation  + " " + bottomRightXBeforeRotation  + " " + bottomRightYBeforeRotation + " " + scale + " " + rotation);
				
				int[] xValues = new int[problemSize];
				int[] yValues = new int[problemSize];
				float[] redInput = new float[problemSize];
				float[] greenInput = new float[problemSize];
				float[] blueInput = new float[problemSize];
				float[] redCanvas = new float[problemSize];
				float[] greenCanvas = new float[problemSize];
				float[] blueCanvas = new float[problemSize];
				float[] opacity = new float[problemSize];
				
				//These variables are to merge two dimensional for into one dimensional array
				int widthX = 0;
				int heightY = 0;
				//This one is important to be the maximum possible value which would describe height of the rect which includes brushstroke.
				int maxHeight = (maximumY - minimumY);
				
				//Now we are ready to collect all data about pixels inside of our bigger square
				for (int x = minimumX; x < maximumX; x++) {
					heightY = 0;
					for (int y = minimumY; y < maximumY; y++) {
						//Get pixel which correspond to the current spot.
			    		PixelRecord surroundPixel = (PixelRecord)data.get(x).get(y);
	
						//Now, our bitmapImage file dosn't have any rotation applied to it
						//However our x and y coords are the coordinates on a rotated brushstroke.
						//Thus we need to convert x and y back to normal bitmap coordinates to get a proper colour form it.
						int origX = (int) Math.round((xCenter + (x - xCenter)*Math.cos(-thetaInRadians) + (y - yCenter)*Math.sin(-thetaInRadians)));
						int origY = (int) Math.round((yCenter - (x - xCenter)*Math.sin(-thetaInRadians) + (y - yCenter)*Math.cos(-thetaInRadians)));
			    		
						//Check if the current coord is in brushstroke, or otherwise there is nothing we can calculate in terms of opacity.
						if (checkIfCoordIsInBrushstroke(origX, origY, topLeftXBeforeRotation, topLeftYBeforeRotation, topRightXBeforeRotation, topRightYBeforeRotation,
														botttomLeftXBeforeRotation, botttomLeftYBeforeRotation, bottomRightXBeforeRotation, bottomRightYBeforeRotation)) {
							
							//Finally to find proper indices inside of a brushstroke, we should subtract topLeft values for coords (x, y).
							surroundPixel.opacity = (float) new Luminance().lum(new Color(bitmapImageScaled.getRGB(Math.abs(origX - topLeftXBeforeRotation), Math.abs(origY - topLeftYBeforeRotation))));
						} else {
							surroundPixel.opacity = 255;
						}
						
						xValues[widthX*maxHeight + heightY] = surroundPixel.x;
			    		yValues[widthX*maxHeight + heightY] = surroundPixel.y;
			    		redInput[widthX*maxHeight + heightY] = (int)surroundPixel.redInput;
			    		greenInput[widthX*maxHeight + heightY] = (int)surroundPixel.greenInput;
			    		blueInput[widthX*maxHeight + heightY] = (int)surroundPixel.blueInput;
			    		redCanvas[widthX*maxHeight + heightY] = (int)surroundPixel.redCanvas;
			    		greenCanvas[widthX*maxHeight + heightY] = (int)surroundPixel.greenCanvas;
			    		blueCanvas[widthX*maxHeight + heightY] = (int)surroundPixel.blueCanvas;
			    		opacity[widthX*maxHeight + heightY] = surroundPixel.opacity;
						
			    		heightY += 1;
					}
			    	widthX += 1;
				}
				
				//Create records which we will send to GPU.
				MainRecord records = new MainRecord();
				records.problemSize = problemSize;
				CudaInterop.getInstance().problemSize = problemSize;
				
				//Honestly, I am a little bit ashamed of this solution, but I simply couldn't figure out how to send a plain float value to CUDA kernel
				//So I had to wrap all the values into arrays.
				float[] luminanceValues = new float[problemSize]; Arrays.fill(luminanceValues, luminance);
				float[] ERCValues = new float[problemSize]; Arrays.fill(ERCValues, ERC);
				float[] mean5x5Values = new float[problemSize]; Arrays.fill(mean5x5Values, mean5x5);
				float[] mean7x7Values = new float[problemSize]; Arrays.fill(mean7x7Values, mean7x7);
				float[] mean9x9Values = new float[problemSize]; Arrays.fill(mean9x9Values, mean9x9);
				float[] mean11x11Values = new float[problemSize]; Arrays.fill(mean11x11Values, mean11x11);
				float[] mean13x13Values = new float[problemSize]; Arrays.fill(mean13x13Values, mean13x13);
				float[] std5x5Values = new float[problemSize]; Arrays.fill(std5x5Values, std5x5);
				float[] std7x7Values = new float[problemSize]; Arrays.fill(std7x7Values, std7x7);
				float[] std9x9Values = new float[problemSize]; Arrays.fill(std9x9Values, std9x9);
				float[] std11x11Values = new float[problemSize]; Arrays.fill(std11x11Values, std11x11);
				float[] std13x13Values = new float[problemSize]; Arrays.fill(std13x13Values, std13x13);
				float[] min5x5Values = new float[problemSize]; Arrays.fill(min5x5Values, min5x5);
				float[] min7x7Values = new float[problemSize]; Arrays.fill(min7x7Values, min7x7);
				float[] min9x9Values = new float[problemSize]; Arrays.fill(min9x9Values, min9x9);
				float[] min11x11Values = new float[problemSize]; Arrays.fill(min11x11Values, min11x11);
				float[] min13x13Values = new float[problemSize]; Arrays.fill(min13x13Values, min13x13);
				float[] max5x5Values = new float[problemSize]; Arrays.fill(max5x5Values, max5x5);
				float[] max7x7Values = new float[problemSize]; Arrays.fill(max7x7Values, max7x7);
				float[] max9x9Values = new float[problemSize]; Arrays.fill(max9x9Values, max9x9);
				float[] max11x11Values = new float[problemSize]; Arrays.fill(max11x11Values, max11x11);
				float[] max13x13Values = new float[problemSize]; Arrays.fill(max13x13Values, max13x13);
				
				records.init(xValues, yValues, redInput, greenInput, blueInput, redCanvas, greenCanvas, blueCanvas, opacity, 
						luminanceValues, ERCValues,
						mean5x5Values, std5x5Values, min5x5Values, max5x5Values,
						mean7x7Values, std7x7Values, min7x7Values, max7x7Values,
						mean9x9Values, std9x9Values, min9x9Values, max9x9Values,
						mean11x11Values, std11x11Values, min11x11Values, max11x11Values,
						mean13x13Values, std13x13Values, min13x13Values, max13x13Values);
				
				//Calculate all pixels
				evalGPU(state, ind, this, ((VectorSpeciesCGP)ind.species).numInputs, records, isItBigImage);
				
				//Wait until CUda will finish the task. It will also return into applyRGB function
				JCuda.cudaDeviceSynchronize();
				
	    	} else {
	    		
	    		//Now we are ready to collect all data about pixels inside of our bigger square
				for (int x = minimumX; x < maximumX; x++) {
					for (int y = minimumY; y < maximumY; y++) {
						//Get pixel which correspond to the current spot.
			    		PixelRecord surroundPixel = (PixelRecord)data.get(x).get(y);
			    		
			    		//Now, our bitmapImage file dosn't have any rotation applied to it
						//However our x and y coords are the coordinates on a rotated brushstroke.
						//Thus we need to convert x and y back to normal bitmap coordinates to get a proper colour form it.
						int origX = (int) Math.round((xCenter + (x - xCenter)*Math.cos(-thetaInRadians) + (y - yCenter)*Math.sin(-thetaInRadians)));
						int origY = (int) Math.round((yCenter - (x - xCenter)*Math.sin(-thetaInRadians) + (y - yCenter)*Math.cos(-thetaInRadians)));
			    		
						//Check if the current coord is in brushstroke, or otherwise there is nothing we can calculate in terms of opacity.
						if (checkIfCoordIsInBrushstroke(origX, origY, topLeftXBeforeRotation, topLeftYBeforeRotation, topRightXBeforeRotation, topRightYBeforeRotation,
														botttomLeftXBeforeRotation, botttomLeftYBeforeRotation, bottomRightXBeforeRotation, bottomRightYBeforeRotation)) {
							
							//Finally to find proper indices inside of a brushstroke, we should subtract topLeft values for coords (x, y).
							surroundPixel.opacity = (float) new Luminance().lum(new Color(bitmapImageScaled.getRGB(Math.abs(origX - topLeftXBeforeRotation), Math.abs(origY - topLeftYBeforeRotation))));
						} else {
							surroundPixel.opacity = 255;
						}
						
						if (surroundPixel.opacity == 255) {
							continue;
						}
	
			    		Float[] inputs = getInputs(surroundPixel, ((VectorSpeciesCGP)ind.species).numInputs);
			    		
			    		Float[] result = eval(state, threadnum, inputs, ind);
	
		            	//Read results
		        		int redOutput = (int)Math.abs(result[3])%256;
		        		int greenOutput = (int)Math.abs(result[4])%256;
		        		int blueOutput = (int)Math.abs(result[5])%256;
		        		
		        		//Set results to become values for canvas RGB
		        		((PixelRecord)(data.get(x).get(y))).redCanvas = redOutput;
		        		((PixelRecord)(data.get(x).get(y))).greenCanvas = greenOutput;
		        		((PixelRecord)(data.get(x).get(y))).blueCanvas = blueOutput;
		        		
		        		((PixelRecord)(data.get(x).get(y))).haveBeenColoured = true;
		        		numberOfPaintedPixels += 1;
		        		
		        		//Apply coloring to the pixel on the canvas image
		         		int col = (redOutput << 16) | (greenOutput << 8) | blueOutput;
		         		(isItBigImage ? problemData.canvasBigImage : problemData.canvasImage).setRGB(x, y, col);
						
					}
				}
	    	}
		}
    	
    }
    
    /*
     * This function checks either a provided (x, y) coordinate lies in a brushstroke.
     */
    public boolean checkIfCoordIsInBrushstroke(int x, int y, int topLeftXBeforeRotation, int topLeftYBeforeRotation, int topRightXBeforeRotation, int topRightYBeforeRotation,
			int botttomLeftXBeforeRotation, int botttomLeftYBeforeRotation, int bottomRightXBeforeRotation, int bottomRightYBeforeRotation) {
    	boolean result = false;
    	
    	//We check this by looking that the (x, y) coord is at the same time bigger by both x and y than the minimum x and y,
    	//and less by both x and y than maximum x and y.
    	
    	if (x > topLeftXBeforeRotation && y > topLeftYBeforeRotation && x < topRightXBeforeRotation && y > topRightYBeforeRotation
    			&& x > botttomLeftXBeforeRotation && y < botttomLeftYBeforeRotation && x < bottomRightXBeforeRotation && y < bottomRightYBeforeRotation) {
    		result = true;
    	}
    	
    	return result;
    }
    
	public void applyRGB(EvolutionState state, Individual ind, KernelOutputData kernelResults, boolean isBig) {
    	OutputData kernelOutput = (OutputData)kernelResults;
    	
    	float[] results = kernelOutput.output;
    	
    	int index = 0;
    	
    	for (int x = minimumX; x < maximumX; x++) {
			for (int y = minimumY; y < maximumY; y++) {
    			float red = results[index];
    			float green = results[index + 1];
    			float blue = results[index + 2];

    			if (red != -1 && green != -1 && blue != -1) {
	            	//Read results
	        		int redOutput = (int)Math.abs(red)%256;
	        		int greenOutput = (int)Math.abs(green)%256;
	        		int blueOutput = (int)Math.abs(blue)%256;
	        		
	        		//Set results to become values for canvas RGB
	        		((PixelRecord)((isBig ? problemData.dataBig : problemData.data).get(x).get(y))).redCanvas = redOutput;
	        		((PixelRecord)((isBig ? problemData.dataBig : problemData.data).get(x).get(y))).greenCanvas = greenOutput;
	        		((PixelRecord)((isBig ? problemData.dataBig : problemData.data).get(x).get(y))).blueCanvas = blueOutput;
	        		
	        		((PixelRecord)((isBig ? problemData.dataBig : problemData.data).get(x).get(y))).haveBeenColoured = true;
	        		numberOfPaintedPixels += 1;
	        		
	        		//Apply coloring to the pixel on the canvas image
	         		int col = (redOutput << 16) | (greenOutput << 8) | blueOutput;
	         		(isBig ? problemData.canvasBigImage : problemData.canvasImage).setRGB(x, y, col);
    			}
         		
         		index += 3;
        	}
    	}
	}
	
	 public void calculateLuminanceValueForGraphLanguage(boolean isItBigImage) {
	    	BufferedImage imageWeWorkWith = isItBigImage ? this.problemData.inputBigImage : this.problemData.inputImage;
	    	Luminance lum = new Luminance();
	    	float total = 0;
	    	for (int x = 0; x < imageWeWorkWith.getWidth(); x++) {
	    		for (int y = 0; y < imageWeWorkWith.getHeight(); y++) {
	    			total += lum.lum(new Color(imageWeWorkWith.getRGB(x, y)));
	    		}
	    	}
	    	luminance = total;
	    }
	    
	    //k is how bg Kxk area is going to be
	    public void calculateAllNeccessaryFieldsForGraphLanguage(int k, PixelRecord finalPixel, boolean isItBigImage) {
	    	BufferedImage imageWeWorkWith = isItBigImage ? this.problemData.canvasBigImage : this.problemData.canvasImage;
	    	float min = Float.MAX_VALUE; // for min
	    	float max = Float.MIN_VALUE; //for max
	    	float total = 0; //for mean
	    	float variance = 0;//for std
	    	Luminance lum = new Luminance();

	    	int x = finalPixel.x;
	        int y = finalPixel.y;
	        
	        int halfPixelsFromK = (k % 2 == 0) ? k/2 : (k+1)/2;
	        
	        if ((x - halfPixelsFromK) < halfPixelsFromK)	{ x = halfPixelsFromK; }
	        if (x > (imageWeWorkWith.getWidth() - halfPixelsFromK)) { x = imageWeWorkWith.getWidth() - halfPixelsFromK; }
	        if ((y - halfPixelsFromK) < halfPixelsFromK)	{ y = halfPixelsFromK; }
	        if (y > (imageWeWorkWith.getHeight() - halfPixelsFromK)) { y = imageWeWorkWith.getHeight() - halfPixelsFromK; }
	        
	        for(int i = x-halfPixelsFromK; i < x + halfPixelsFromK; i++) {
	        	for(int j = y - halfPixelsFromK; j < y+halfPixelsFromK; j++) {
	    			float value = (float) lum.lum(new Color(imageWeWorkWith.getRGB(i, j)));
	    			
	    			total += value;
	    			if (min > value) {
	    				min = value;
	    			}
	    			if (max < value) {
	    				max = value;
	    			}
	        	}
	        }
	        
	        float mean = total / (float)(k * k);
	                
	        for(int i = x-halfPixelsFromK; i < x+halfPixelsFromK; i++) {
	        	for(int j = y-halfPixelsFromK; j < y+halfPixelsFromK; j++) {
	    			float value = (float) lum.lum(new Color(imageWeWorkWith.getRGB(i, j)));
	        		variance +=  Math.pow(value - mean,2);
	        	}
	        }
	        
	        float std = (float) Math.sqrt(variance/(k * k));
	        
	        if (k == 5) {
	        	mean5x5 = mean;
	        	min5x5 = min;
	        	max5x5 = max;
	        	std5x5 = std;
	        } else if (k == 7) {
	        	mean7x7 = mean;
	        	min7x7 = min;
	        	max7x7 = max;
	        	std7x7 = std;
	        } else if (k == 9) {
	        	mean9x9 = mean;
	        	min9x9 = min;
	        	max9x9 = max;
	        	std9x9 = std;
	        } else if (k == 11) {
	        	mean11x11 = mean;
	        	min11x11 = min;
	        	max11x11 = max;
	        	std11x11 = std;
	        } else if (k == 13) {
	        	mean13x13 = mean;
	        	min13x13 = min;
	        	max13x13 = max;
	        	std13x13 = std;
	        }
	    }
    
    public static void main(String[] args) {
    	String[] arguments = new String[] {"-file", "src/main/NPRwithCGPonGPUs.params"};
//    	CudaPrimitive2D.usePitchedMemory(true);	// Do use the pitched memory FIXME pitch is not supported
//    	unless the kernel takes the pitch values as well!
    	ec.Evolve.main(arguments);
    }
}

