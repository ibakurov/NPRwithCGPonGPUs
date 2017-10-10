package fitness;

import java.awt.image.BufferedImage;

import eval.Response;

public class Skewness {
	
	//Link to formula: http://www.itl.nist.gov/div898/handbook/eda/section3/eda35b.htm
	public double skewness(BufferedImage image) {
		BufferedImage grayScaleImage = new Luminance().turnImageToGrayScale(image);
		double SD = new DeviationFromNormal(0).getRaw(new Response(image), image, image);
//		double mean = new Mean(0).getRaw(new Response(image), image, image);
		
		//Sum of subtractions in power to 3
		double sub = 0;
		for (int x = 0; x < image.getWidth(); x++) {
			for (int y = 0; y < image.getHeight(); y++) {
				int rgb = image.getRGB(x, y);
				int rgbInGrayScale = grayScaleImage.getRGB(x, y);
				sub += Math.pow(Math.abs(rgb - rgbInGrayScale), 3);
			}
		}
		
		int numberOfPixels = image.getWidth() * image.getHeight();
		double adjustmentForSampleSize = Math.sqrt(numberOfPixels * (numberOfPixels - 1)) / (numberOfPixels - 1);
		double result = adjustmentForSampleSize * ((sub / numberOfPixels) / Math.pow(SD, 3));
		
		return result;
	}
	
}
