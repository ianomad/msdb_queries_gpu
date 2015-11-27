#include <string>
#include <fstream>
#include <iostream>

using namespace std;

int main(int arc, char **argv) {
	
	string fileName = argv[1];
	ifstream stream(fileName);

	cout << "Reading file... " << fileName << endl;

	string head;
	int num;
	double val;

	int rows = 0;
	int heads = 0;

	while(stream >> head) {
		if(head.compare("HEAD") == 0) {
			heads++;
			printf("HEAD: %d\n", heads);
		}
	}

	printf("Rows: %d\n", rows);

	return 0;
}