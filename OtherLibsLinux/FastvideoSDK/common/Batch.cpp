/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
with this source code for terms and conditions that govern your use of
this software. Any use, reproduction, disclosure, or distribution of
this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#include "Batch.hpp"

template<typename T>
Batch<T>::Batch(int batchSize) {
	this->batchSize = batchSize;
	this->filledItem = batchSize;

	values = new T[batchSize];
	isFree = true;
}

template<typename T>
Batch<T>::~Batch() {
	delete[] values;
}

template<typename T>
T* Batch<T>::At(int i) {
	if (i < batchSize)
		return &values[i];
	return nullptr;
}

template<typename T>
unsigned int Batch<T>::GetSize() const {
	return batchSize;
}

template<typename T>
unsigned int Batch<T>::GetFilledItem() const {
	return this->filledItem;
}

template<typename T>
void Batch<T>::SetFilltedItem(unsigned int count) {
	this->filledItem = count;
}
