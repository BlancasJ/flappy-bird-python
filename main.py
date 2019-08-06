import pygame
import os

from game_bird import play, BG_IMG, WIN_WIDTH, WIN_HEIGHT
from game_neat import train

TITLE_FONT = pygame.font.SysFont("comicsans", 70)
OPTION_FONT = pygame.font.SysFont("comicsans", 40)


def draw_menu(win):
    win.blit(BG_IMG, (0, 0))

    title = TITLE_FONT.render("Flappy Bird", 1, (255, 255, 255))
    win.blit(title, (WIN_WIDTH // 2 - title.get_width() // 2, 120))

    option1 = OPTION_FONT.render("1 - Manual", 1, (255, 255, 255))
    option2 = OPTION_FONT.render("2 - NEAT (AI)", 1, (255, 255, 255))
    option_quit = OPTION_FONT.render("ESC - Quit", 1, (255, 255, 255))

    win.blit(option1, (WIN_WIDTH // 2 - option1.get_width() // 2, 350))
    win.blit(option2, (WIN_WIDTH // 2 - option2.get_width() // 2, 420))
    win.blit(option_quit, (WIN_WIDTH // 2 - option_quit.get_width() // 2, 490))

    hint = OPTION_FONT.render("(ESC in-game returns here)", 1, (200, 200, 200))
    win.blit(hint, (WIN_WIDTH // 2 - hint.get_width() // 2, 620))

    pygame.display.update()


def menu_loop(win, config_path):
    pygame.display.set_caption("Flappy Bird — Menu")
    clock = pygame.time.Clock()

    while True:
        clock.tick(30)
        draw_menu(win)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return
                if event.key == pygame.K_1:
                    result = play(win)
                    if result == "quit":
                        return
                if event.key == pygame.K_2:
                    result = train(win, config_path)
                    if result == "quit":
                        return


def main():
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    menu_loop(win, config_path)
    pygame.quit()


if __name__ == "__main__":
    main()
