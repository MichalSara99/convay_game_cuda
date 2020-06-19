#include<iostream>
#include<string>

#include<cuda_runtime.h>
#include<device_launch_parameters.h>

#include"convay_game.h"

int main(int argc, char const* argv[]) {


	std::size_t const boardSize = 20;
	std::size_t totalSize = boardSize * boardSize;
	std::size_t const gens = 700;


	ConvayGame game{ boardSize,gens };
	game.play();



	std::cin.get();
	std::cin.get();
	return 0;
}