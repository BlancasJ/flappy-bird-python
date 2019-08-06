import pygame
import neat
import os

from game_bird import Bird, Pipe, Base, BG_IMG, STAT_FONT, WIN_WIDTH, WIN_HEIGHT

GEN = 0


class StopTraining(Exception):
    def __init__(self, reason):
        self.reason = reason


def draw_window(win, birds, pipes, base, score, gen, alive):
    win.blit(BG_IMG, (0, 0))
    for pipe in pipes:
        pipe.draw(win)

    score_text = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(score_text, (WIN_WIDTH - 10 - score_text.get_width(), 10))

    gen_text = STAT_FONT.render("Gen: " + str(gen), 1, (255, 255, 255))
    win.blit(gen_text, (10, 10))

    alive_text = STAT_FONT.render("Alive: " + str(alive), 1, (255, 255, 255))
    win.blit(alive_text, (10, 60))

    base.draw(win)
    for bird in birds:
        bird.draw(win)

    pygame.display.update()


def _make_eval_genomes(win):
    def eval_genomes(genomes, config):
        global GEN
        GEN += 1

        nets = []
        ge = []
        birds = []

        for _, genome in genomes:
            genome.fitness = 0
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            nets.append(net)
            birds.append(Bird(230, 350))
            ge.append(genome)

        base = Base(730)
        pipes = [Pipe(600)]
        pygame.display.set_caption("Flappy Bird — NEAT")
        clock = pygame.time.Clock()
        score = 0

        while len(birds) > 0:
            clock.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise StopTraining("quit")
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    raise StopTraining("menu")

            pipe_ind = 0
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1

            for x, bird in enumerate(birds):
                ge[x].fitness += 0.1
                bird.move()

                output = nets[x].activate((
                    bird.y,
                    abs(bird.y - pipes[pipe_ind].height),
                    abs(bird.y - pipes[pipe_ind].bottom),
                ))

                if output[0] > 0.5:
                    bird.jump()

            add_pipe = False
            rem = []
            for pipe in pipes:
                for x, bird in enumerate(birds):
                    if pipe.collide(bird):
                        ge[x].fitness -= 1
                        birds.pop(x)
                        nets.pop(x)
                        ge.pop(x)

                if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                    rem.append(pipe)

                if not pipe.passed and birds and pipe.x < birds[0].x:
                    pipe.passed = True
                    add_pipe = True

                pipe.move()

            if add_pipe:
                score += 1
                for g in ge:
                    g.fitness += 5
                pipes.append(Pipe(600))

            for r in rem:
                pipes.remove(r)

            for x, bird in enumerate(birds):
                if bird.y + bird.img.get_height() >= 730 or bird.y < 0:
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)

            base.move()
            draw_window(win, birds, pipes, base, score, GEN, len(birds))

            if score > 50:
                break

    return eval_genomes


def train(win, config_path):
    global GEN
    GEN = 0

    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    try:
        winner = p.run(_make_eval_genomes(win), 50)
        print("\nBest genome:\n{!s}".format(winner))
        return "menu"
    except StopTraining as e:
        print("\nTraining stopped by user.")
        return e.reason


def run(config_path):
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    train(win, config_path)
    pygame.quit()


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)
