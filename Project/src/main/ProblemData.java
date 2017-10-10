package main;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

import ec.cgp.Record;

public class ProblemData {

	/** storage for records */
	List<ArrayList<Record>> data = new ArrayList<ArrayList<Record>>();
	List<ArrayList<Record>> dataBig = new ArrayList<ArrayList<Record>>();

	static public String titleOfImage = "../Assets/Cherry";
	static public String titleOfCHISTQImage = "../Assets/Cherry";
	static public String titleOfFolder = "1.Experiment";
	
    public BufferedImage inputImage;				// whole image to work with
    public BufferedImage inputBigImage;				// whole image to work with
    public BufferedImage targetCHISTQImage;				// whole image to work with
    public BufferedImage targetCHISTQBigImage;				// whole image to work with
    public BufferedImage canvasImage;				// whole image to work with
    public BufferedImage canvasBigImage;				// whole image to work with
	
	public void init() {
		// Read and parse images
        try {
        	this.inputImage = (BufferedImage) ImageIO.read(new File(titleOfImage+"Big.png"));
		} catch (IOException e) {}
        
        try {
        	this.inputBigImage = (BufferedImage) ImageIO.read(new File(titleOfImage+"Big.png"));
		} catch (IOException e) {}
        
        try {
        	this.targetCHISTQImage = (BufferedImage) ImageIO.read(new File(titleOfCHISTQImage+"Big.png"));
		} catch (IOException e) {}
        
        try {
        	this.targetCHISTQBigImage = (BufferedImage) ImageIO.read(new File(titleOfCHISTQImage+"Big.png"));
		} catch (IOException e) {}
        
        //Fully black
        this.canvasImage = new BufferedImage(this.inputImage.getWidth(), this.inputImage.getHeight(), BufferedImage.TYPE_INT_RGB);
        //Fully white
    	for (int i = 0; i < canvasImage.getWidth(); i++) {
        	for (int j = 0; j < canvasImage.getHeight(); j++) {
        		int col = (255 << 16) | (255 << 8) | 255;
        		canvasImage. setRGB(i, j, col);
        	}
    	}
    	
    	//Fully black
        this.canvasBigImage = new BufferedImage(this.inputBigImage.getWidth(), this.inputBigImage.getHeight(), BufferedImage.TYPE_INT_RGB);
        //Fully white
    	for (int i = 0; i < canvasBigImage.getWidth(); i++) {
        	for (int j = 0; j < canvasBigImage.getHeight(); j++) {
        		int col = (255 << 16) | (255 << 8) | 255;
        		canvasBigImage. setRGB(i, j, col);
        	}
    	}
    	
    	
    	
	    if (data.isEmpty()) {
	    	
	    	data.addAll(makeRecord(this.inputImage, this.canvasImage));
			dataBig.addAll(makeRecord(this.inputBigImage, this.canvasBigImage));
		}
	}
	
	/**
	 * Interpret each pixel from an image into a Record instance.
	 */
	ArrayList<ArrayList<Record>> makeRecord(BufferedImage inputImg, BufferedImage canvasImg) {
		ArrayList<ArrayList<Record>> allRecords = new ArrayList<ArrayList<Record>>();
	
   	 	assert inputImg == null || canvasImg == null : "Input image and canvas image must be assigned in setup process!";
   	 	assert(inputImg.getWidth() == canvasImg.getWidth() && inputImg.getHeight() == canvasImg.getHeight()) : "Width or height of input image and canvas image are different";
		for (int i=0; i < inputImg.getWidth(); i++) {
			ArrayList<Record> lineRecord = new ArrayList<Record>();
			for (int j=0; j < inputImg.getHeight(); j++) {
				PixelRecord r = new PixelRecord();
				PixelData pixelInput = getPixelAtCurrentCoordinates(inputImg, i, j);
				PixelData pixelCanvas = getPixelAtCurrentCoordinates(canvasImg, i, j);
				
				r.init(i, j, (float)pixelInput.r, (float)pixelInput.g, (float)pixelInput.b, (float)pixelCanvas.r, (float)pixelCanvas.g, (float)pixelCanvas.b, 255.f);
				
				lineRecord.add(r);
			}
			allRecords.add(lineRecord);
		}
    			
		return allRecords;
	}

	  /**
     * Methods that gets the RGB values of pixel in the current coordinates
     * @return PixelData
     */
    public PixelData getPixelAtCurrentCoordinates(BufferedImage img, int x, int y) {
    	// Getting pixel colour by coordinates x and y 
		int clr =  img.getRGB(x, y);
		float  red   = (clr & 0x00ff0000) >> 16;
		float  green = (clr & 0x0000ff00) >> 8;
		float  blue  =  clr & 0x000000ff;
		  
		return new PixelData(x, y, red, green, blue);	
    }
}
