# style_transfer_cs1470
Implementation of "A Neural Algorithm of Artistic Style" for CSCI1470.



To run:
1. Make sure you have downloaded the pretrained VGG-19 Network: http://www.vlfeat.org/matconvnet/pretrained/
2. Put the network within the directory where main.py is
3. The command to run the program is:

python main.py <Path to content image> <Path to style image>

Example:

python main.py images/neckarfront3.jpg artwork/gogh.jpg


If you run into any issues involving missing packages/modules, you probably have to run pip install <name of package>
