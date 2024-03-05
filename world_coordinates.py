import numpy as np

#The starting point is in the upper-left corner of the field
#the y-axis pointing downwards and the x-axis pointing to the right.

world_coordinates=[
                    #x,y,z
                   [0,0,0],
                   [0,1.370,0],
                   [0,5.485,0],
                   [0,9.600,0],
                   [0,10.970,0],

                   [5.485,1.370,0],
                   [5.485,5.485,0],
                   [5.485,9.600,0],

                   [11.885,0,0],
                   [11.885,1.370,0],
                   [11.885,5.485,0],
                   [11.885,9.600,0],
                   [11.885,10.970,0],

                   [18.285,1.370,0],
                   [18.285,5.485,0],
                   [18.285,9.600,0],

                   [23.770,0,0],
                   [23.770,1.370,0],
                   [23.770,5.485,0],
                   [23.770,9.600,0],
                   [23.770,10.970,0],

                   [11.885,0,1.2],
                   [11.885,1.370,1.2],
                   [11.885,5.485,1.2],
                   [11.885,9.600,1.2],
                   [11.885,10.970,1.2],


                   ]
world_coordinates = {
    "world_coordinates": world_coordinates
}

np.save("./world_coordinates.npy", world_coordinates)