add_executable(test_app test.cpp)

set(test_app_TARGETS test_app)

# Propagate back to the parent scope
set(test_app_TARGETS ${test_app_TARGETS} PARENT_SCOPE)