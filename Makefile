# Compilers
GCC = gcc-13

# Libraries
LIBS = -lm

# Directories
SRC_DIR = src
BUILD_DIR = build
DEBUG_DIR = $(BUILD_DIR)/debug
DEBUG_OBJ_DIR = $(DEBUG_DIR)/obj
RELEASE_DIR = $(BUILD_DIR)/release
RELEASE_OBJ_DIR = $(RELEASE_DIR)/obj

# Flags
COMMON_FLAGS = -Wall -Wextra -std=c17 -I$(SRC_DIR)
DEBUG_FLAGS = $(COMMON_FLAGS) -g -O0
RELEASE_FLAGS = $(COMMON_FLAGS) -DNDEBUG -O2

# Sources
SRCS = $(shell find $(SRC_DIR) -type f -name '*.c')

# Objects
DEBUG_OBJS = $(patsubst $(SRC_DIR)/%.c, $(DEBUG_OBJ_DIR)/%.o, $(SRCS))
RELEASE_OBJS = $(patsubst $(SRC_DIR)/%.c, $(RELEASE_OBJ_DIR)/%.o, $(SRCS))

# Dependencies
DEBUG_DEPS = $(DEBUG_OBJS:%.o=%.d)
RELEASE_DEPS = $(RELEASE_OBJS:%.o=%.d)

# Binaries
DEBUG_BIN = $(DEBUG_DIR)/sinew_mind
RELEASE_BIN = $(RELEASE_DIR)/sinew_mind

# Targets
.PHONY: all run clean debug \
	debug-run release release-run

all: debug

run: debug-run

clean:
	@rm -rf $(DEBUG_DIR) $(RELEASE_DIR)
	@echo "All build artifacts cleaned!"

debug: $(DEBUG_BIN)
	@echo "Debug build complete: $<"

debug-run: debug
	@"$(DEBUG_BIN)"

release: $(RELEASE_BIN)
	@echo "Release build complete: $<"

release-run: release
	@"$(RELEASE_BIN)"

# Directory Creation
$(DEBUG_DIR) $(RELEASE_DIR):
	@mkdir -p $@

# Compilation
.DELETE_ON_ERROR:
$(DEBUG_OBJS): $(DEBUG_OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(@D)
	$(GCC) $(DEBUG_FLAGS) -MMD -MP -c $< -o $@

$(RELEASE_OBJS): $(RELEASE_OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(@D)
	$(GCC) $(RELEASE_FLAGS) -MMD -MP -c $< -o $@

# Linking
$(DEBUG_BIN): $(DEBUG_OBJS) | $(DEBUG_DIR)
	$(GCC) $(DEBUG_FLAGS) $^ $(LIBS) -o $@

$(RELEASE_BIN): $(RELEASE_OBJS) | $(RELEASE_DIR)
	$(GCC) $(RELEASE_FLAGS) $^ $(LIBS) -o $@

# Includes
-include $(DEBUG_DEPS)
-include $(RELEASE_DEPS)