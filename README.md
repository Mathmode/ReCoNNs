This project contains the code for the use of Regularity Conforming Neural Network (ReCoNNs) architectures for solving PDEs. For a detailed explaination of the methodology behind the code, the preprint of the corresponding article is available at arXiv:2405.14110

The code covers four test examples: A 1D problem with a jump in the derivative, a 2D problem with jumps in the gradient across an interface, the L-shape problem that admits a singularity at the re-entrant corner, and a case of interior material vertices, showing both of the last two types of singularities. 

Each example admits its own "Main_X.py" file (Main_1D, Main_jump, Main_L, Main_4_Materials, respectively). The relevant architectures are found in the SRC.Architectures_X files, and the corresponding loss functions in SRC.Loss_X. SRC.Postprocessing is used to plot the 2D solutions once obtained. 
