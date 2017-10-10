package fitness;

import java.awt.Color;
import java.awt.image.BufferedImage;

import ec.cgp.representation.PixelData;

public class Luminance {

	public BufferedImage turnImageToGrayScale(BufferedImage image) {
		BufferedImage grayScaleImage = new BufferedImage(image.getWidth(), image.getHeight(), image.getType());
		for (int x = 0; x < image.getWidth(); x++) {
			for (int y = 0; y < image.getHeight(); y++) {
				grayScaleImage.setRGB(x, y, toGray(getColourAtCoordinates(image, x, y)).getRGB());
			}
		}
		return grayScaleImage;
	}
	
	public Color getColourAtCoordinates(BufferedImage image, int x, int y) {
		// Getting pixel colour by coordinates x and y
		return new Color(image.getRGB(x, y));
	}
	
	// return the monochrome luminance of given color
    public double lum(Color color) {
        int r = color.getRed();
        int g = color.getGreen();
        int b = color.getBlue();
        return 0.299*r + 0.587*g + 0.114*b;
    }

    // return a gray version of this Color
    public Color toGray(Color color) {
        int y = (int) (Math.round(lum(color)));   // round to nearest int
        Color gray = new Color(y, y, y);
        return gray;
    }
    
    public double distance(BufferedImage canvas, BufferedImage input) {
    	double total = 0;
    	for (int i = 0; i < input.getWidth(); i++) {
    		for (int j = 0; j < input.getHeight(); j++) {
    			total += Math.abs(getPixelAtCurrentCoordinates(canvas, i, j).r - getPixelAtCurrentCoordinates(input, i, j).r);
    		}
    	}
    	return total;// / (input.getWidth() * input.getHeight());
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
