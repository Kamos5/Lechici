import pygame
import sys
import subprocess
import os

# Initialize Pygame
pygame.init()

# Screen settings
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Main Menu")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Font
try:
    font = pygame.font.SysFont("arial", 40)
except:
    font = pygame.font.Font(None, 40)  # Fallback to default font

# Menu options
options = ["New Game", "Load Game", "Options", "Quit"]
selected_option = 0  # Tracks the currently selected option for keyboard navigation

# Button settings
BUTTON_WIDTH = 200
BUTTON_HEIGHT = 50
BUTTON_MARGIN = 20
TOTAL_BUTTON_HEIGHT = (BUTTON_HEIGHT + BUTTON_MARGIN) * len(options) - BUTTON_MARGIN
START_Y = (SCREEN_HEIGHT - TOTAL_BUTTON_HEIGHT) // 2  # Center vertically

# Placeholder message settings
message = ""
message_timer = 0
MESSAGE_DURATION = 2000  # 2 seconds in milliseconds

def draw_menu():
    screen.fill(BLACK)  # Clear screen with black background
    buttons = []

    # Draw each button
    for i, option in enumerate(options):
        button_y = START_Y + i * (BUTTON_HEIGHT + BUTTON_MARGIN)
        button_rect = pygame.Rect(SCREEN_WIDTH // 2 - BUTTON_WIDTH // 2, button_y, BUTTON_WIDTH, BUTTON_HEIGHT)
        buttons.append(button_rect)

        # Check if mouse is hovering or option is selected via keyboard
        mouse_pos = pygame.mouse.get_pos()
        is_hovered = button_rect.collidepoint(mouse_pos)
        is_selected = (i == selected_option)

        # Draw button background (blue if hovered or selected, white otherwise)
        pygame.draw.rect(screen, BLUE if (is_hovered or is_selected) else WHITE, button_rect)
        pygame.draw.rect(screen, RED, button_rect, 2)  # Red border

        # Draw button text
        text_surface = font.render(option, True, BLACK)
        text_rect = text_surface.get_rect(center=button_rect.center)
        screen.blit(text_surface, text_rect)

    # Draw placeholder message if active
    if message and pygame.time.get_ticks() < message_timer:
        message_surface = font.render(message, True, WHITE)
        message_rect = message_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 50))
        screen.blit(message_surface, message_rect)

    return buttons

def main_menu():
    global selected_option, message, message_timer
    clock = pygame.time.Clock()
    running = True

    while running:
        buttons = draw_menu()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # Handle mouse clicks
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                for i, button in enumerate(buttons):
                    if button.collidepoint(mouse_pos):
                        selected_option = i
                        if i == 0:  # New Game
                            try:
                                subprocess.run([sys.executable, "main80.py"], check=True)
                            except FileNotFoundError:
                                message = "Error: main80.py not found!"
                                message_timer = pygame.time.get_ticks() + MESSAGE_DURATION
                            except Exception as e:
                                message = f"Error: {str(e)}"
                                message_timer = pygame.time.get_ticks() + MESSAGE_DURATION
                        elif i == 1:  # Load Game
                            message = "Load Game: Feature not implemented"
                            message_timer = pygame.time.get_ticks() + MESSAGE_DURATION
                        elif i == 2:  # Options
                            message = "Options: Feature not implemented"
                            message_timer = pygame.time.get_ticks() + MESSAGE_DURATION
                        elif i == 3:  # Quit
                            pygame.quit()
                            sys.exit()

            # Handle keyboard input
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected_option = (selected_option - 1) % len(options)
                elif event.key == pygame.K_DOWN:
                    selected_option = (selected_option + 1) % len(options)
                elif event.key == pygame.K_RETURN:
                    if selected_option == 0:  # New Game
                        try:
                            subprocess.run([sys.executable, "main80.py"], check=True)
                        except FileNotFoundError:
                            message = "Error: main80.py not found!"
                            message_timer = pygame.time.get_ticks() + MESSAGE_DURATION
                        except Exception as e:
                            message = f"Error: {str(e)}"
                            message_timer = pygame.time.get_ticks() + MESSAGE_DURATION
                    elif selected_option == 1:  # Load Game
                        message = "Load Game: Feature not implemented"
                        message_timer = pygame.time.get_ticks() + MESSAGE_DURATION
                    elif selected_option == 2:  # Options
                        message = "Options: Feature not implemented"
                        message_timer = pygame.time.get_ticks() + MESSAGE_DURATION
                    elif selected_option == 3:  # Quit
                        pygame.quit()
                        sys.exit()

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main_menu()