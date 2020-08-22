# /usr/bin/env python3

import numpy
import curses
from curses import wrapper
import time
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit

from pycuda.compiler import SourceModule

BLOCKSIZE = 32
GPU_NITER = 10

row2str = lambda row: ''.join(['0' if c != 0 else ' ' for c in row])  

def calc_next_world_gpu(world, next_world):
    height, width = world.shape
    ## CUDAカーネルを定義
    mod = SourceModule("""
    __global__ void get_next_world(int *world, int *nextWorld, int height, int width){
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        
        const int index = y * width + x;
        int current_value;
        int next_value;
        if (x >= width) {
            return;
        }
        if (y >= height) {
            return;
        }
        current_value = world[index];

        int numlive = 0;    
        numlive += world[((y - 1) % height ) * width + ((x - 1) % width)];
        numlive += world[((y - 1) % height ) * width + ( x      % width)]; 
        numlive += world[((y - 1) % height ) * width + ((x + 1) % width)]; 
        numlive += world[( y      % height ) * width + ((x - 1) % width)];
        numlive += world[( y      % height ) * width + ((x + 1) % width)];
        numlive += world[((y + 1) % height ) * width + ((x - 1) % width)];
        numlive += world[((y + 1) % height ) * width + ( x      % width)]; 
        numlive += world[((y + 1) % height ) * width + ((x + 1) % width)];

        if (current_value == 0 && numlive == 3){
            next_value = 1;
        }else if (current_value == 1 && numlive >= 2 && numlive <= 3){
            next_value = 1;
        }else{
            next_value = 0;
        }
        nextWorld[index] = next_value; 
    }
    """)
    set_next_cell_value_GPU = mod.get_function("get_next_world")
    block = (BLOCKSIZE, BLOCKSIZE, 1)
    grid = ((width + block[0] - 1) // block[0], (height + block[1] - 1) // block[1])
    set_next_cell_value_GPU(cuda.In(world),cuda.Out(next_world),numpy.int32(height), numpy.int32(width), block = block, grid = grid)

def print_world(stdscr, world, generation, elapsed):
    height, width = world.shape
    for y in range(height):
        row = world[y]
        stdscr.addstr(y, 0, row2str(row))
    stdscr.addstr(height, 0, "Generation: %06d, Elapsed: %.6f[sec]" % (generation, elapsed / generation), curses.A_REVERSE)
    stdscr.refresh()

def game_of_life(stdscr, height, width):
    world = numpy.random.randint(2, size=(height, width), dtype=numpy.int32)
    next_world = numpy.empty((height, width), dtype=numpy.int32)

    elapsed = 0.0
    generation = 0
    while True:
        generation += 1
        print_world(stdscr, world, generation, elapsed)
        start_time = time.time()
        calc_next_world_gpu(world, next_world)   
        duration = time.time()-start_time
        elapsed += duration
        world, next_world = next_world, world

def main(stdscr):
    stdscr.clear()
    stdscr.nodelay(True)
    scr_height, scr_width = stdscr.getmaxyx()
    game_of_life(stdscr, scr_height - 1, scr_width)

if __name__ == '__main__':
    curses.wrapper(main)
