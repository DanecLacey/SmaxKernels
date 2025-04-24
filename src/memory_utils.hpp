#ifndef MEMORY_UTILS_HPP
#define MEMORY_UTILS_HPP

#ifndef ALIGNMENT
#define ALIGNMENT 64
#endif

// Overwrite heap allocators to use posix_memalign

void *aligned_malloc(size_t bytesize)
{
	int errorCode;
	void *ptr;

	errorCode = posix_memalign(&ptr, ALIGNMENT, bytesize);

	if (errorCode)
	{
		if (errorCode == EINVAL)
		{
			fprintf(stderr,
					"Error: Alignment parameter is not a power of two\n");
			exit(EXIT_FAILURE);
		}
		if (errorCode == ENOMEM)
		{
			fprintf(stderr,
					"Error: Insufficient memory to fulfill the request\n");
			exit(EXIT_FAILURE);
		}
	}

	if (ptr == NULL)
	{
		fprintf(stderr, "Error: posix_memalign failed!\n");
		exit(EXIT_FAILURE);
	}

	return ptr;
}

// Overload new and delete for alignement
void *operator new(size_t bytesize)
{
	// printf("Overloading new operator with size: %lu\n", bytesize);
	int errorCode;
	void *ptr;
	errorCode = posix_memalign(&ptr, ALIGNMENT, bytesize);

	if (errorCode)
	{
		if (errorCode == EINVAL)
		{
			fprintf(stderr,
					"Error: Alignment parameter is not a power of two\n");
			exit(EXIT_FAILURE);
		}
		if (errorCode == ENOMEM)
		{
			fprintf(stderr,
					"Error: Insufficient memory to fulfill the request for space\n");
			exit(EXIT_FAILURE);
		}
	}

	if (ptr == NULL)
	{
		fprintf(stderr, "Error: posix_memalign failed!\n");
		exit(EXIT_FAILURE);
	}

	return ptr;
}

void operator delete(void *p)
{
	// printf("Overloading delete operator\n");
	free(p);
}

// Generic accessor for casting a void* to a reference of type T
template <typename T>
inline T &as(void *ptr)
{
	return *static_cast<T *>(ptr);
}

template <typename T>
inline const T &as(const void *ptr)
{
	return *static_cast<const T *>(ptr);
}

#endif // MEMORY_UTILS_HPP
