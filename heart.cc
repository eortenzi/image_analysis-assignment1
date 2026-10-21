#include <iostream>
#include <memory>
#include <stdexcept>
using namespace std;

void printHelensHeart(string &string) {

}

int main() {
    try {
        string heart = "Helen Heart's";
        string *pointerToHelensHeart = &heart;

        cout << heart << endl;
        cout << *pointerToHelensHeart << endl;
    }
    catch (const out_of_range &exception) {
        cerr << exception.what() << endl;
    }
    catch (...) {
    }
    return 0;
}

