#include "resnet_c/resnet.h"

void timestamp();

int main() {
  float data[224][224][3];
  Resnet__resnet50_out _out;
  timestamp();
  Resnet__resnet50_step(data, &_out);
  timestamp();
}
