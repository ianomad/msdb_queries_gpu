#include <string>
#include <fstream>
#include <iostream>

using namespace std;

int main(int arc, char **argv) {
	
	string fileName = argv[1];
	ifstream stream(fileName.c_str());

	cout << "Reading file... " << fileName << endl;

	string head;
	int num;
	int frame;
	double val;

	int rows = 0;
	int heads = 0;
	int atoms = 0;

	while(stream >> head) {
		if(head.compare("HEAD") == 0) {
			heads++;
			stream >> frame;
			stream >> num;
			printf("HEAD: %d %d %d %d\n", heads, frame, num, atoms);
			rows++;
			atoms = 0;
		} else if(head.compare("ATOM") == 0) {
			rows++;
			atoms++;
		}
	}

	printf("Heads: %d\n", heads);
	printf("Rows: %d\n", rows);
	printf("Atoms: %d\n", atoms);

	return 0;
}